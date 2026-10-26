# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton solver boundary for a temporal NeRD model."""

import math

import newton
import torch
import warp as wp
from physicsnemo_robotics.dynamics import (
    CompositeFeatureEncoder,
    ControlFieldEncoder,
    FieldNormalizer,
    LearnedDynamicsSolver,
    TrajectoryFieldSpec,
    TrajectorySchema,
    TransitionAdapter,
    TransitionSpec,
)
from physicsnemo_robotics.newton import (
    NewtonExecutionSpec,
    NewtonModelSignature,
    NewtonSimulationSpec,
    NewtonSolverSignature,
    NewtonTimeSpec,
)
from physicsnemo_robotics.newton.simulation import collision_pipeline_configuration

from isaaclab.utils import string_to_callable

from .nerd import JointDynamics, model_signature
from .newton_manager import NewtonManager


class SolverNeRD(LearnedDynamicsSolver):
    """Native checkpoint and effort/identity policy around PhysicsNeMo's resident solver."""

    def __init__(self, model, checkpoint, *, compile=True, temporal="auto", collision_args=None):
        if not model.device.is_cuda:
            raise ValueError("The native NeRD solver currently requires CUDA.")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != "isaaclab-nerd-v3":
            raise ValueError(
                "Refit the recorded transitions to a v3 checkpoint with canonical PhysicsNeMo heading inputs."
            )
        self.dynamics = JointDynamics(saved["dynamics"], model)
        if model_signature(model, contact_conditioned=bool(self.dynamics.max_contacts)) != saved["dynamics"]["model"]:
            raise ValueError("NeRD checkpoint physical model differs from the live scene.")
        network = (
            string_to_callable(saved["network_type"])(
                self.dynamics.input_dim,
                self.dynamics.codec.prediction_shape[0],
                **saved["network_cfg"],
                **self.dynamics.contact_model_cfg,
            )
            .to(model.device.alias)
            .eval()
        )
        network.load_state_dict(saved["model"])
        network.requires_grad_(False)
        history = 1 + sum(span - 1 for span in saved["network_cfg"]["temporal_attention_spans"])
        collision = newton.CollisionPipeline(model, **(collision_args or {})) if self.dynamics.max_contacts else None
        # This is the live deployment contract. The native checkpoint identity was
        # checked above; no teacher solver or portable artifact is inferred here.
        features = [ControlFieldEncoder("joint_f", per_world_shape=(self.dynamics.control_dim,))]
        fields = [
            TrajectoryFieldSpec(
                "control/joint_f", "transition", dtype="float32", trailing_shape=(self.dynamics.control_dim,)
            )
        ]
        if self.dynamics.gravity is not None:
            features.append(self.dynamics.gravity)
        if self.dynamics.contact_adapter is not None:
            features.append(self.dynamics.contact_adapter)
            fields.append(self.dynamics.contact_adapter.field_spec)
        if self.dynamics.max_contacts:
            fields.append(self.dynamics.contact_encoder.field_spec)
        identity = NewtonModelSignature(
            topology="isaaclab-uniform-joint-worlds",
            static_fingerprint=self.dynamics.descriptor["model"],
            newton_version=newton.__version__,
        )
        simulation = NewtonSimulationSpec(
            model=identity,
            solver=NewtonSolverSignature("isaaclab_newton.physics.nerd_manager.SolverNeRD"),
            time=NewtonTimeSpec(solver_dt=self.dynamics.descriptor["frame_dt"], substeps=1),
        )
        execution = NewtonExecutionSpec(
            stepper_implementation="isaaclab_newton.physics.nerd_manager.NewtonNeRDManager",
            collision_pipeline_implementation=(
                f"{type(collision).__module__}.{type(collision).__qualname__}" if collision is not None else None
            ),
            resolved_collision_configuration=(
                collision_pipeline_configuration(collision) if collision is not None else {}
            ),
            collide_on_reset=False,
            collide_each_substep=bool(self.dynamics.max_contacts),
            clear_forces=True,
            requires_grad=False,
        )
        schema = TrajectorySchema.from_specs(simulation, execution, self.dynamics.codec, fields=fields)
        adapter = TransitionAdapter(
            schema,
            self.dynamics.codec,
            TransitionSpec(history=history),
            features=CompositeFeatureEncoder(*features),
            contact_features=self.dynamics.contact_encoder if self.dynamics.max_contacts else None,
        )

        # Only models promising repeated-frame per-world resets can use cold streaming.
        if temporal == "auto":
            temporal = "stream" if getattr(network, "supports_repeated_frame_reset", False) else "full_window"
        super().__init__(
            model,
            network,
            adapter,
            normalizer=FieldNormalizer.from_dict(saved["normalizer"]),
            input_normalizer_key="input",
            prediction_normalizer_key="target",
            contact_normalizer_key="contact" if self.dynamics.max_contacts else None,
            model_signature=adapter.schema.simulation.model,
            continuity="off",
            temporal=temporal,
            compile=compile,
        )
        # Compile once on disposable state; the manager later captures the entire physics frame.
        state, control = model.state(), model.control()
        contacts = None
        if collision is not None:
            contacts = collision.contacts()
            collision.collide(state, contacts)
        with torch.no_grad():
            self.step(state, state, control, contacts, self.frame_dt)
            self.reset(state)

    @torch.no_grad()
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        self.dynamics.check_inputs(state_in)
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        if self.dynamics.max_contacts:
            with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.model.device))):
                torch._assert_async(
                    (self.dynamics.contact_encoder.last_overflow == 0).all(), "NeRD contact capacity overflowed."
                )

    def notify_model_changed(self, flags):
        # Fixed-base root-pose resets can notify unchanged joint properties. Verify the same
        # physical contract instead of suppressing every notification or mutating a library method.
        if (
            int(flags)
            and model_signature(self.model, contact_conditioned=bool(self.dynamics.max_contacts))
            != self.dynamics.descriptor["model"]
        ):
            raise ValueError("Physical model changes invalidate the trained NeRD checkpoint.")


class NewtonNeRDManager(NewtonManager):
    """Construct learned physics through the same instance lifecycle as other Newton solvers."""

    def _build_solver(self, model, solver_cfg):
        if self._num_substeps != 1 or not solver_cfg.checkpoint:
            raise ValueError("NeRD requires num_substeps=1 and a checkpoint path.")
        if (
            set(model.get_requested_contact_attributes()) & newton.Contacts.EXTENDED_ATTRIBUTES
            or set(model.get_requested_state_attributes()) & newton.State.EXTENDED_ATTRIBUTES
        ):
            raise ValueError("NeRD does not produce extended sensor state or solved contact forces.")
        self._solver = SolverNeRD(
            model,
            solver_cfg.checkpoint,
            compile=solver_cfg.compile,
            temporal=solver_cfg.temporal,
            collision_args=self._collision_cfg.to_pipeline_args() if self._collision_cfg is not None else None,
        )
        if not math.isclose(self._solver_dt, self._solver.frame_dt, rel_tol=1e-9):
            raise ValueError("Set sim.dt to the NeRD checkpoint's physics-step duration.")
        self._use_single_state = True
        self._needs_collision_pipeline = bool(self._solver.dynamics.max_contacts)
        self._supports_rigid_body_force_input = False
        self._supports_contact_sensors = False

    def _capture_graph(self, capture_target):
        # Warp owns graph capture/replay. Isolate its Torch temporaries from the caller's caching
        # allocator and keep this pool alive as long as the captured physics graph can use them.
        self._torch_pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(self._torch_pool, device=self._device):
            return super()._capture_graph(capture_target)

    def clear(self):
        if self._solver is not None:
            self._solver.session.close()
        super().clear()

    def _solver_specific_clear(self):
        self._torch_pool = None
