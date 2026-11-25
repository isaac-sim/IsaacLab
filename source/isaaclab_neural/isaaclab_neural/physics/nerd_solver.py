# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton solver adapter (:class:`NeRDSolver`) wrapping ``newton-neural``'s ``SolverNeRD`` for Isaac Lab."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonSolver
from newton import Model
from newton.solvers import SolverBase, SolverMuJoCo
from newton_neural.models import NeRDBase
from newton_neural.solvers import SolverNeRD, SolverNeRDMuJoCo, load_nerd
from newton_neural.systems import SystemArticulation

from isaaclab.cloner import expand_env_regex_ns

from ..nerd import default_compile

if TYPE_CHECKING:
    from .nerd_solver_cfg import NeRDSolverCfg

logger = logging.getLogger(__name__)

COUPLING_ERROR = (
    "steps a model, which keeps one history per world of the whole simulation and cannot yet run as an "
    "entry of a coupled solver"
)
"""Error suffix returned by ``coupling_error()`` for neural solvers."""

_UNWRITTEN_STATE = {
    "body_qdd": "IMU and PVA sensors",
    "body_parent_f": "joint wrench sensors and robots with enable_joint_wrench",
}
"""Newton state attributes not computed by ``SolverNeRD`` and the Isaac Lab features that read them."""


class NeRDSolver(NewtonSolver):
    """Isaac Lab ``NewtonSolver`` wrapper for ``newton-neural``'s standalone ``SolverNeRD``.

    Uses the ordinary Newton manager and backend. The task declares the model frame as its physics step.
    Key semantics declared to Isaac Lab:

    - Imports scenes with MuJoCo USD schemas so the resulting ``newton.Model`` matches recording.
    - Uses Newton's collision pipeline; contact sensors read penetration-based normal-force estimates.
    - Rejects IMU, PVA, and joint-wrench sensors (accelerations and reaction wrenches are not computed).
    - External body forces have no effect, and each kind of runtime model mutation emits a ``NewtonNeuralWarning``
      once.
    - On CUDA, Isaac Lab captures the step into its physics-step CUDA graph (internal solver graphing disabled);
      any environment state write clears that world's streaming history.
    """

    cfg: NeRDSolverCfg

    solver_class = SolverNeRD
    single_state = True

    @classmethod
    def solver_classes(cls, cfg: NeRDSolverCfg) -> tuple[type[SolverBase], ...]:
        """Include ``SolverMuJoCo`` schemas during scene import so the Newton model matches recording."""
        return (SolverNeRD, SolverMuJoCo)

    @classmethod
    def coupling_error(cls, cfg: NeRDSolverCfg, in_place: bool) -> str:
        """Return why neural solvers cannot run inside Isaac Lab's coupled-solver wrapper."""
        return COUPLING_ERROR

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverNeRD:
        """Build ``SolverNeRD`` for the configured articulation and validate the Newton model."""
        cfg = self.cfg
        device = wp.device_to_torch(model.device)
        nerd = load_nerd(cfg.checkpoint, device)
        require_articulation(nerd)
        requested = model.get_requested_state_attributes()
        unwritten = [f"{name} ({readers})" for name, readers in _UNWRITTEN_STATE.items() if name in requested]
        if unwritten:
            raise ValueError(
                f"the task reads {' and '.join(unwritten)} from the Newton state. A model that predicts the "
                "next state computes no accelerations and no joint reaction forces, so these would stay zero. "
                "Remove what reads them from the scene configuration"
            )
        solver = SolverNeRD(
            model,
            nerd,
            articulation=articulation_pattern(model, cfg.articulation, env_template),
            precision=cfg.precision,
            compile=default_compile(device) if cfg.compile is None else cfg.compile,
            # Disable internal solver graphs because Isaac Lab captures the full physics step itself.
            use_cuda_graph=False,
        )
        check_model(solver)
        logger.info(f"Built {solver}")
        return solver

    def applies_body_forces(self) -> bool:
        """Return ``False`` (neural state-increment models ignore external body wrenches)."""
        return False

    def supports_capture(self) -> bool:
        """Return ``True`` on CUDA only (Warp CPU capture omits PyTorch ops)."""
        return self.native.model.device.is_cuda


def require_articulation(nerd: NeRDBase) -> SystemArticulation:
    """Validate that ``nerd.system`` is a ``SystemArticulation`` and return it."""
    system = nerd.system
    if not isinstance(system, SystemArticulation):
        raise TypeError(
            f"the system of the model is a {type(system).__name__}, and an Isaac Lab task runs on models of "
            "articulated robots, whose system is a SystemArticulation"
        )
    return system


def articulation_pattern(model: Model, articulation: str, env_template: str | None) -> re.Pattern[str]:
    """Compile a regex matching Newton articulation labels at or below prim path ``articulation``."""
    expression = (
        expand_env_regex_ns(articulation) if env_template is None else expand_env_regex_ns(articulation, env_template)
    )
    pattern = re.compile(f"(?:{expression})(?:/.*)?")
    if not any(pattern.fullmatch(label) for label in model.articulation_label):
        raise ValueError(
            f"no articulation of the scene lies at {articulation!r}, the prim path that the solver configuration "
            "gives for the robot. Set its articulation to the prim path of the robot's asset, such as "
            f"env_cfg.scene.robot.prim_path. The first articulations of the scene are {model.articulation_label[:4]}"
        )
    return pattern


def check_model(solver: SolverNeRD | SolverNeRDMuJoCo) -> None:
    """Validate the checkpoint against the bound Newton model, independently of environment actions."""
    model, nerd = solver.model, solver.nerd
    system = nerd.system
    if "isaaclab" not in nerd.metadata:
        raise ValueError("Checkpoint has no Isaac Lab model contract; record and train it with scripts/neural")
    expected = nerd.metadata["isaaclab"]["physical_model"]
    actual = physical_contract(model, solver.binding, free_motion=not nerd.prediction.integrates)
    if actual != expected:
        raise ValueError("NeRD checkpoint physical model differs from the live scene; record and train it again")
    # Target-only models include actuator dynamics; active Newton actuators would convert targets into ignored forces.
    robot_dofs = solver.binding.joint_dof_ids
    if system.includes_actuators and any(
        np.isin(actuator.indices.numpy(), robot_dofs).any() for actuator in model.actuators
    ):
        raise ValueError(
            f"the checkpoint consumes {', '.join(system.spec.control)} and includes actuator dynamics, but Newton "
            "actuators still drive the robot's joints. Use isaaclab.actuators.ImplicitActuatorCfg to pass the "
            "joint targets to the model"
        )


def physical_contract(model, binding, *, free_motion: bool) -> dict:
    """Read the robot properties that a fixed learned model cannot observe as inputs.

    Every recorded world must describe the same robot. World placement and native actuator replacement are
    intentionally excluded; binding validates kinematics and gravity independently.
    """
    bodies = binding.body_ids.numpy()
    dofs = binding.joint_dof_ids
    arrays = {
        "body_mass": model.body_mass.numpy()[bodies],
        "body_inertia": model.body_inertia.numpy()[bodies],
        "body_com": model.body_com.numpy()[bodies],
        "joint_armature": model.joint_armature.numpy()[dofs],
    }
    if free_motion:
        arrays.update(
            joint_target_ke=model.joint_target_ke.numpy()[dofs],
            joint_target_kd=model.joint_target_kd.numpy()[dofs],
        )
    for name, values in arrays.items():
        if not np.allclose(values, values[:1], rtol=1e-6, atol=1e-7):
            raise ValueError(f"NeRD requires uniform {name} across recorded worlds")
    return {name: values[0].tolist() for name, values in arrays.items()}
