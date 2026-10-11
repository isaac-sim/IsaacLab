# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager that advances an Isaac Lab scene with a PhysicsNeMo dynamics artifact."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import Contacts, Control, Model, State
from physicsnemo_robotics.dynamics import DynamicsArtifact, LearnedDynamicsSolver
from physicsnemo_robotics.newton import FrameSpec, NewtonModelSignature

from isaaclab.utils.string import string_to_callable

from ..heading import artifact_codec_factory, artifact_feature_factory
from ..resident_training import _fp32_parallel_nerd_attention
from ..signature import isaaclab_canonical_view
from .state_ownership import validate_effort_control, validate_state_ownership

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext

    from .physicsnemo_manager_cfg import PhysicsNeMoSolverCfg


class NewtonPhysicsNeMoManager(NewtonManager):
    """Newton manager whose solver is a PhysicsNeMo Robotics learned-dynamics artifact.

    The artifact predicts the next generalized state of every environment from its recent history and
    the applied joint efforts, and Newton forward kinematics rebuilds the body state. The manager:

    * validates the live scene against the artifact identity, frame duration, state ownership, and
      effort-only actuation before the first step, and rejects contact sensors and extended state
      outputs it cannot produce;
    * fails on external body wrenches, which are not model inputs;
    * runs eagerly, because the network call cannot be captured in a Warp CUDA graph;
    * forwards Isaac Lab environment resets through the inherited solver reset hook, so each reset
      environment restarts its model history;
    * accepts the joint-property notifications Isaac Lab raises when a task rewrites an unchanged
      fixed-base root pose, and rejects every other model change.
    """

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize Newton and disable contact sensors before the scene is created."""
        super().initialize(sim_context)
        NewtonManager._supports_contact_sensors = False

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: PhysicsNeMoSolverCfg) -> LearnedDynamicsSolver:
        """Construct the learned solver without reading or changing manager state."""
        return _create_learned_solver(model, solver_cfg, frame_dt=None)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: PhysicsNeMoSolverCfg) -> None:
        """Build the learned solver for the active environment and publish the canonical solver slots."""
        NewtonManager._solver = None
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._supports_rigid_body_force_input = False
        NewtonManager._supports_contact_sensors = False

        num_envs = cls.get_num_envs()
        if num_envs != model.world_count:
            raise ValueError(
                f"Learned dynamics needs one Newton world per environment, got {model.world_count} worlds for"
                f" {num_envs} environments."
            )
        if cls._num_substeps != 1:
            raise ValueError(
                f"Learned dynamics advances one frame per physics step; set num_substeps=1, got {cls._num_substeps}."
            )
        if NewtonManager._report_contacts:
            raise NotImplementedError("Learned dynamics does not report Newton contact forces.")

        # Property writes queued during scene initialization are part of the finalized model that the
        # artifact is validated against below, so they must not reach the learned solver as changes.
        NewtonManager._model_changes = set()
        solver = _create_learned_solver(model, solver_cfg, frame_dt=cls._solver_dt)
        _accept_unchanged_root_pose_writes(solver, model)
        NewtonManager._solver = solver
        NewtonManager._needs_collision_pipeline = solver.operator.adapter.requires_contacts

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Advance one learned frame, failing on external body wrenches that the model cannot represent."""
        if state_0.body_f is not None:
            # Checked on the device, so the step never waits for the host.
            torch._assert_async(
                ~wp.to_torch(state_0.body_f).any(),
                "External body wrenches in State.body_f are not inputs of the learned dynamics model.",
            )
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Learned inference runs eager Torch code that Warp graph capture cannot enclose."""
        return False


def _create_learned_solver(
    model: Model, solver_cfg: PhysicsNeMoSolverCfg, *, frame_dt: float | None
) -> LearnedDynamicsSolver:
    """Load the configured artifact and build a learned solver checked against the live model.

    Args:
        model: Finalized top-level Newton model of the scene.
        solver_cfg: Learned solver configuration.
        frame_dt: Physics step the solver will be called with [s], or ``None`` to skip the check.
    """
    if isinstance(model, newton.solvers.experimental.coupled.ModelView):
        raise NotImplementedError("Learned dynamics cannot run as an entry of a Newton coupled solver.")
    for kind, requested, extended in (
        ("state", model.get_requested_state_attributes(), newton.State.EXTENDED_ATTRIBUTES),
        ("contact", model.get_requested_contact_attributes(), newton.Contacts.EXTENDED_ATTRIBUTES),
    ):
        unsupported = sorted(set(requested) & extended)
        if unsupported:
            raise NotImplementedError(
                f"Learned dynamics does not produce the extended Newton {kind} attributes {unsupported} requested by"
                " the scene's sensors."
            )

    artifact = DynamicsArtifact.load(solver_cfg.artifact_path, model_factory=_resolve(solver_cfg.model_factory))
    simulation = artifact.manifest.schema.simulation
    if simulation.frame != FrameSpec():
        raise ValueError("Isaac Lab learned dynamics requires an artifact trained in the canonical Newton frame.")
    if frame_dt is not None and not math.isclose(frame_dt, simulation.time.frame_dt, rel_tol=1e-9):
        raise ValueError(
            f"The physics step {frame_dt} s must equal the artifact frame duration {simulation.time.frame_dt} s."
        )
    if artifact.requires_modeled_system_signature:
        raise NotImplementedError("Portable modeled-contact artifacts are not supported by Isaac Lab deployment.")

    # Rebuilt features such as fixed contact probes fingerprint model geometry, which must not depend on
    # the number of environments.
    with isaaclab_canonical_view(model):
        solver = artifact.as_solver(
            model,
            model_signature=NewtonModelSignature.from_model(model),
            codec_factory=_resolve(solver_cfg.codec_factory) or artifact_codec_factory,
            feature_factory=_resolve(solver_cfg.feature_factory) or artifact_feature_factory,
            device=str(model.device),
            continuity=solver_cfg.continuity,
        )
    validate_state_ownership(model, solver.operator.adapter.codec)
    validate_effort_control(model)
    if solver_cfg.inference_dtype == "float16":
        _half_precision_projections(solver.operator.model)
    return solver


def _half_precision_projections(network: torch.nn.Module) -> None:
    """Run the network under FP16 autocast and return FP32 predictions, so states stay in FP32.

    Attention scores stay in FP32, because trained ParallelNeRD logits can exceed the FP16 range.
    """
    for name in ("forward", "predict_last"):
        method = getattr(network, name, None)
        if not callable(method):
            continue

        def half_precision(*args, _method=method, **kwargs):
            with torch.autocast("cuda", dtype=torch.float16), _fp32_parallel_nerd_attention():
                output = _method(*args, **kwargs)
            return output.float() if isinstance(output, torch.Tensor) else output

        setattr(network, name, half_precision)


def _accept_unchanged_root_pose_writes(solver: LearnedDynamicsSolver, model: Model) -> None:
    """Drop joint-property notifications while the joint anchor transforms equal the trained ones.

    Newton stores fixed-base root poses in the joint anchors ``model.joint_X_p`` and ``model.joint_X_c``,
    and Isaac Lab notifies the solver with :attr:`newton.ModelFlags.JOINT_PROPERTIES` after every root-pose
    write, including resets that write the unchanged default pose. The learned solver rejects every model
    change, so only notifications that leave the trained anchors unchanged are dropped. ``model.joint_q``,
    which the flag also covers, holds initial coordinates that the learned solver never reads.
    """
    anchors = (model.joint_X_p, model.joint_X_c)
    trained_anchors = tuple(wp.to_torch(anchor).clone() for anchor in anchors)
    reject_model_change = solver.notify_model_changed

    def notify_model_changed(flags: newton.ModelFlags | int) -> None:
        if int(flags) == int(newton.ModelFlags.JOINT_PROPERTIES) and all(
            torch.equal(wp.to_torch(anchor), trained) for anchor, trained in zip(anchors, trained_anchors)
        ):
            return
        reject_model_change(flags)

    solver.notify_model_changed = notify_model_changed


def _resolve(factory: Callable | str | None) -> Callable | None:
    """Resolve an optional factory configured as a callable or an import string."""
    return string_to_callable(factory) if isinstance(factory, str) else factory
