# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton solver boundary for a temporal NeRD model."""

import math

import newton
import torch
import warp as wp

from isaaclab.utils import string_to_callable

from .nerd import JointDynamics, model_signature
from .nerd_forces import _ForceReconstruction
from .newton_manager import NewtonManager


@wp.kernel
def _write_joint_state(
    values: wp.array2d(dtype=float), q: wp.array(dtype=float), qd: wp.array(dtype=float), nq: int, nv: int
):
    world, column = wp.tid()
    if column < nq:
        q[world * nq + column] = values[world, column]
    else:
        qd[world * nv + column - nq] = values[world, column]


class SolverNeRD(newton.solvers.SolverBase):
    """Fixed-storage history, resets and inference; all Python decisions are construction-time decisions."""

    def __init__(self, model, checkpoint, *, compile=True, reconstruct_forces=False):
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("The native NeRD solver currently requires CUDA.")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != "isaaclab-nerd-v2":
            raise ValueError("Expected a native Isaac Lab NeRD checkpoint.")
        self.dynamics = JointDynamics(saved["dynamics"], model)
        if model_signature(model, contact_conditioned=bool(self.dynamics.max_contacts)) != saved["dynamics"]["model"]:
            raise ValueError("NeRD checkpoint physical model differs from the live scene.")
        self.frame_dt = saved["dynamics"]["frame_dt"]
        self.reconstruction = _ForceReconstruction(model, self.frame_dt) if reconstruct_forces else None
        device = model.device.alias
        self.network = (
            string_to_callable(saved["network_type"])(
                self.dynamics.input_dim,
                self.dynamics.codec.prediction_shape[0],
                **saved["network_cfg"],
                **self.dynamics.contact_model_cfg,
            )
            .to(device)
            .eval()
        )
        self.network.load_state_dict(saved["model"])
        self.network.requires_grad_(False)
        self.mean = saved["input_mean"].to(device)
        self.scale = saved["input_scale"].to(device)
        self.output_mean = saved["target_mean"].to(device)
        self.output_scale = saved["target_scale"].to(device)
        history = 1 + sum(span - 1 for span in saved["network_cfg"]["temporal_attention_spans"])
        self.history = torch.zeros(model.world_count, history, self.dynamics.input_dim, device=device)
        self.contact_history = None
        self.contact_mean = self.contact_scale = None
        if self.dynamics.max_contacts:
            self.contact_history = torch.zeros(model.world_count, history, *self.dynamics.contact_shape, device=device)
            self.contact_mean, self.contact_scale = saved["contact_mean"].to(device), saved["contact_scale"].to(device)
        self.pending_reset = torch.ones(model.world_count, dtype=torch.bool, device=device)
        self.predict = torch.compile(self.network.predict_last) if compile else self.network.predict_last
        # Warm model kernels and compilation outside the host's graph capture. No physical state advances.
        with torch.no_grad(), torch.cuda.stream(wp.stream_to_torch(wp.get_stream(model.device))):
            if self.contact_history is None:
                self.predict(self.history)
            else:
                self.predict(
                    self.history,
                    self.dynamics.contact_tokens(self.contact_history, self.contact_mean, self.contact_scale),
                )

    @torch.no_grad()
    def reset(self, state, *, world_mask=None, flags=0):
        with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.model.device))):
            if world_mask is None:
                self.pending_reset.fill_(True)
            else:
                self.pending_reset.logical_or_(wp.to_torch(world_mask)[: self.model.world_count])
            if self.reconstruction is not None:
                self.reconstruction.reset(state, world_mask)

    @torch.no_grad()
    def step(self, state_in, state_out, control, contacts, dt):
        if not math.isclose(dt, self.frame_dt, rel_tol=1e-9):
            raise ValueError(f"NeRD frame is {self.frame_dt}s, received {dt}s.")
        with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.model.device))):
            self.dynamics.check_inputs(state_in)
            action = wp.to_torch(control.joint_f)
            state = self.dynamics.codec.read(state_in)
            contact_values = self.dynamics.contacts(state_in, contacts)
            features = self.dynamics.inputs(state, action.reshape(self.model.world_count, -1), contact_values)
            token = (features - self.mean) / self.scale
            shifted = torch.cat((self.history[:, 1:], token[:, None]), dim=1)
            self.history.copy_(torch.where(self.pending_reset[:, None, None], token[:, None], shifted))
            if self.contact_history is None:
                prediction = self.predict(self.history)
            else:
                shifted = torch.cat((self.contact_history[:, 1:], contact_values[:, None]), dim=1)
                self.contact_history.copy_(
                    torch.where(self.pending_reset[:, None, None, None], contact_values[:, None], shifted)
                )
                prediction = self.predict(
                    self.history,
                    self.dynamics.contact_tokens(self.contact_history, self.contact_mean, self.contact_scale),
                )
            self.pending_reset.zero_()
            delta = prediction * self.output_scale + self.output_mean
            prediction = self.dynamics.codec.delta_to_state(state, delta)
            torch._assert_async(torch.isfinite(prediction).all(), "NeRD produced a nonfinite state.")
            layout = self.dynamics.codec.layout
            if self.reconstruction is not None:
                self.reconstruction.prepare(state_in, control, contacts)
            wp.launch(
                _write_joint_state,
                dim=prediction.shape,
                inputs=[
                    wp.from_torch(prediction.contiguous()),
                    state_out.joint_q,
                    state_out.joint_qd,
                    layout.dof_q,
                    layout.dof_qd,
                ],
                device=self.model.device,
            )
            newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)
            if self.reconstruction is not None:
                self.reconstruction.update(state_out)

    def get_max_contact_count(self):
        """Size the shared Newton contact buffer for force reporting."""
        return 0 if self.reconstruction is None else self.reconstruction.solver.get_max_contact_count()

    def update_contacts(self, contacts, state=None):
        """Publish reconstructed forces on the same Newton collision rows used by the prediction."""
        if self.reconstruction is None:
            raise ValueError("Enable NeRDSolverCfg.reconstruct_forces to use contact sensors.")
        self.reconstruction.solver._populate_contact_observables(contacts, contacts.force, preserve_geometry=True)

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
        requested_contacts = set(model.get_requested_contact_attributes()) & newton.Contacts.EXTENDED_ATTRIBUTES
        requested_state = set(model.get_requested_state_attributes()) & newton.State.EXTENDED_ATTRIBUTES
        if requested_contacts - {"force"} or requested_state - {"body_parent_f"}:
            raise ValueError("NeRD force reconstruction supports contact forces and incoming joint wrenches only.")
        if (requested_contacts or requested_state) and not solver_cfg.reconstruct_forces:
            raise ValueError("Enable NeRDSolverCfg.reconstruct_forces to use contact or joint-wrench sensors.")
        self._solver = SolverNeRD(
            model, solver_cfg.checkpoint, compile=solver_cfg.compile, reconstruct_forces=solver_cfg.reconstruct_forces
        )
        if not math.isclose(self._solver_dt, self._solver.frame_dt, rel_tol=1e-9):
            raise ValueError("Set sim.dt to the NeRD checkpoint's physics-step duration.")
        self._use_single_state = True
        self._needs_collision_pipeline = bool(self._solver.dynamics.max_contacts) or solver_cfg.reconstruct_forces
        self._supports_rigid_body_force_input = False
        self._supports_contact_sensors = solver_cfg.reconstruct_forces

    def _capture_graph(self, capture_target):
        # Warp owns graph capture/replay. Isolate its Torch temporaries from the caller's caching
        # allocator and keep this pool alive as long as the captured physics graph can use them.
        self._torch_pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(self._torch_pool, device=self._device):
            return super()._capture_graph(capture_target)

    def _solver_specific_clear(self):
        self._torch_pool = None
