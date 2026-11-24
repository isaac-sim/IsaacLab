# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager (:class:`NewtonNeRDManager`) that steps a state-increment model once per environment step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonManager
from newton_neural.solvers import load_nerd, same_duration

from .nerd_solver import require_articulation
from .nerd_solver_cfg import NeRDSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonCfg


class NewtonNeRDManager(NewtonManager):
    """Newton manager of a task whose solver is a state-increment model.

    A numerical solver takes several small steps per environment step. A model takes one, of the frame it was trained
    on, and that frame is one environment step of the task, ``sim.dt * decimation``. This manager makes the frame its
    physics step and takes one solver step per environment step, so actions, actuators, collision detection and
    sensors run once per environment step. The task keeps its own ``sim.dt`` and ``decimation``.

    Select it with ``class_type`` of a :class:`~isaaclab_newton.physics.NewtonCfg` whose solver configuration is a
    :class:`~isaaclab_neural.physics.NeRDSolverCfg` and whose ``num_substeps`` is 1.

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
        if not isinstance(cfg.solver_cfg, NeRDSolverCfg):
            raise ValueError(
                f"NewtonNeRDManager steps a model, and the solver of its NewtonCfg is configured by a "
                f"{type(cfg.solver_cfg).__name__}. Give it a NeRDSolverCfg"
            )
        if cfg.num_substeps != 1:
            raise ValueError(
                f"a model advances one environment step per solver step, got num_substeps {cfg.num_substeps}. "
                "Set num_substeps of the NewtonCfg to 1"
            )
        # The frame is needed before the backend exists; cache the loaded model so the first solver build reuses it.
        nerd = load_nerd(cfg.solver_cfg.checkpoint, "cpu")
        require_articulation(nerd)
        self._frame_dt: float = nerd.frame_dt
        self._initial_nerd: tuple[str, object] | None = (cfg.solver_cfg.checkpoint, nerd)

    def pop_initial_nerd(self, checkpoint: str):
        """Return and clear the startup-loaded model if ``checkpoint`` matches, avoiding a duplicate disk read."""
        cached = self._initial_nerd
        self._initial_nerd = None
        if cached is not None and cached[0] == checkpoint:
            return cached[1]
        return None

    def get_physics_dt(self) -> float:
        """Duration of one physics step [s], which is the frame of the model."""
        return self._frame_dt

    def set_decimation(self, decimation: int) -> None:
        """Check that the frame of the model is one environment step of ``decimation`` physics steps of ``sim.dt``.

        The backend keeps taking one physics step per :meth:`step`.

        Raises:
            ValueError: If ``sim.dt * decimation`` differs from the frame of the model.
        """
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
        return True

    def step(self) -> None:
        """Advance the physics by one frame of the model, which is one environment step.

        Raises:
            RuntimeError: If the simulation plays and no environment declared its decimation with
                :meth:`set_decimation`.
        """
        sim = self._sim
        if not self._frame_checked and sim is not None and sim.is_playing():
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
