# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton construction owner for standalone and MuJoCo-coupled learned dynamics."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from newton import Model
from newton_neural.solvers import load_nerd, same_duration

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionManager
from isaaclab.scene import InteractiveScene
from isaaclab.utils.string import resolve_matching_names_values

from .mjwarp_nerd_solver import MJWarpNeRDSolver
from .mjwarp_nerd_solver_cfg import MJWarpNeRDSolverCfg
from .nerd_solver import NeRDSolver, check_task, physical_contract, require_articulation
from .nerd_solver_cfg import NeRDSolverCfg


class NewtonNeRDManager(NewtonManager):
    """Load one learned model and supply it explicitly through Newton's normal solver construction.

    A standalone state-increment model takes one step of the frame it was trained
    on, and that frame is one environment step of the task, ``sim.dt * decimation``. This manager makes the frame its
    physics step and takes one solver step per environment step, so actions, actuators, collision detection and
    sensors run once per environment step. The task keeps its own ``sim.dt`` and ``decimation``.

    For standalone dynamics, select :class:`~isaaclab_neural.physics.NeRDSolverCfg` and ``num_substeps=1``.
    For free-motion coupling, select :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg`; ordinary Newton
    timing and decimation apply, and the checkpoint frame must match each MuJoCo Warp solver substep.

    .. code-block:: python

        replace(
            newton_mjwarp,
            class_type="isaaclab_neural.physics.newton_nerd_manager:NewtonNeRDManager",
            solver_cfg=NeRDSolverCfg(checkpoint="anymal_c.pt", articulation="{ENV_REGEX_NS}/Robot"),
            num_substeps=1,
        )

    The manager relies on the environment for two things. The environment declares its decimation with
    :meth:`set_decimation`, and it steps the simulation once per environment step when :meth:`handles_decimation` says
    so. The manager-based and the direct environments of :mod:`isaaclab.envs` do both. A loop that calls
    ``sim.step()`` once per ``sim.dt`` would advance one frame per call, so :meth:`step` raises while the simulation
    plays and no decimation is declared.

    ``sim.dt`` is not the step of the physics under this manager. :meth:`get_physics_dt` returns the frame, while
    ``sim.get_physics_dt()`` still returns ``sim.dt``, which code that integrates between two physics steps reads.
    """

    def __init__(self, cfg: NewtonCfg):
        # Set before the base class clears its state. A hard reset keeps the declared decimation.
        self._frame_checked = False
        super().__init__(cfg)
        if not isinstance(cfg.solver_cfg, NeRDSolverCfg | MJWarpNeRDSolverCfg):
            raise ValueError(
                f"NewtonNeRDManager steps a model, and the solver of its NewtonCfg is configured by a "
                f"{type(cfg.solver_cfg).__name__}. Give it a NeRDSolverCfg or MJWarpNeRDSolverCfg"
            )
        if isinstance(cfg.solver_cfg, NeRDSolverCfg) and cfg.num_substeps != 1:
            raise ValueError(
                f"a model advances one environment step per solver step, got num_substeps {cfg.num_substeps}. "
                "Set num_substeps of the NewtonCfg to 1"
            )
        self.nerd = load_nerd(cfg.solver_cfg.checkpoint, "cpu")
        require_articulation(self.nerd)
        if "isaaclab" not in self.nerd.metadata:
            raise ValueError("Checkpoint has no Isaac Lab model contract; record and train it with scripts/neural")
        self._frame_dt = self.nerd.frame_dt
        self._integrates = self.nerd.prediction.integrates
        if isinstance(cfg.solver_cfg, NeRDSolverCfg) != self._integrates:
            raise ValueError("The checkpoint prediction does not match the configured NeRD solver kind")

    def create_solver(
        self,
        cfg: NeRDSolverCfg | MJWarpNeRDSolverCfg,
        model: Model,
        mode: wp.DeterministicMode,
        env_template: str | None,
    ) -> NeRDSolver | MJWarpNeRDSolver:
        """Pass resolved resources to the adapter and validate the checkpoint before stepping."""
        if self._integrates:
            adapter = NeRDSolver(
                cfg, model, mode, env_template, nerd=self.nerd, contact_sensors=bool(self.backend.contact_sensors)
            )
        else:
            adapter = MJWarpNeRDSolver(cfg, model, mode, env_template, nerd=self.nerd)
        check_task(adapter.native, self.backend, self._sim.cfg.use_newton_actuators)
        expected = self.nerd.metadata["isaaclab"]["physical_model"]
        actual = physical_contract(model, adapter.native.binding, free_motion=not self._integrates)
        if actual != expected:
            raise ValueError("NeRD checkpoint physical model differs from the live scene; record and train it again")
        return adapter

    def get_physics_dt(self) -> float:
        """Duration of one physics step [s], which is the frame of the model."""
        return self._frame_dt if self._integrates else super().get_physics_dt()

    def handle_actions(self, actions: ActionManager, scene: InteractiveScene, *, enabled: bool = True) -> bool:
        """Validate declared action clipping against the checkpoint before registering action callbacks."""
        bounds = self.nerd.metadata["isaaclab"]["action_bounds"]
        if bounds:
            for name in actions.active_terms:
                term = actions.get_term(name)
                if not isinstance(term, JointPositionAction):
                    raise ValueError("A bounded NeRD checkpoint requires joint-position actions")
                if term.cfg.clip is None:
                    raise ValueError("The NeRD checkpoint requires the recorded joint-target clipping bounds")
                ids, names, limits = resolve_matching_names_values(term.cfg.clip, term._joint_names)
                if len(ids) != len(term._joint_names):
                    raise ValueError("Every NeRD joint target needs an explicit clipping bound")
                for joint, (low, high) in zip(names, limits, strict=True):
                    index = bounds["joints"].index(joint)
                    if low < bounds["low"][index] - 1e-6 or high > bounds["high"][index] + 1e-6:
                        raise ValueError(f"Joint {joint} clipping exceeds the NeRD checkpoint's recorded bounds")
        return super().handle_actions(actions, scene, enabled=enabled)

    def set_decimation(self, decimation: int) -> None:
        """Check that the frame of the model is one environment step of ``decimation`` physics steps of ``sim.dt``.

        The backend keeps taking one physics step per :meth:`step`.

        Raises:
            ValueError: If ``sim.dt * decimation`` differs from the frame of the model.
        """
        if not self._integrates:
            super().set_decimation(decimation)
            return
        decimation = max(1, decimation)
        sim_dt = self._sim.cfg.dt
        if not same_duration(sim_dt * decimation, self._frame_dt):
            raise ValueError(
                f"the model at {self._cfg.solver_cfg.checkpoint} steps frames of {self._frame_dt} s, and one "
                f"environment step of the task lasts {sim_dt * decimation} s, which is sim.dt {sim_dt} s times "
                f"decimation {decimation}. A model that replaces the solver advances one environment step per frame. "
                "Train a model on frames of that duration"
            )
        self._frame_checked = True

    def handles_decimation(self) -> bool:
        """Return ``True``, since one :meth:`step` advances the whole environment step."""
        return self._integrates or super().handles_decimation()

    def step(self) -> None:
        """Advance the physics by one frame of the model, which is one environment step.

        Raises:
            RuntimeError: If the simulation plays and no environment declared its decimation with
                :meth:`set_decimation`.
        """
        sim = self._sim
        if self._integrates and not self._frame_checked and sim is not None and sim.is_playing():
            raise RuntimeError(
                "a model that replaces the solver advances one environment step per sim.step(), and no environment "
                "declared its decimation to the physics manager. Step the simulation from an environment that calls "
                "set_decimation and honors handles_decimation, as the manager-based and the direct environments of "
                "isaaclab.envs do, or call sim.backend_manager.set_decimation yourself and step once per "
                "environment step. An environment that calls "
                "sim.step() once per sim.dt, such as those of the Warp frontend, would advance decimation frames per "
                "environment step"
            )
        super().step()
