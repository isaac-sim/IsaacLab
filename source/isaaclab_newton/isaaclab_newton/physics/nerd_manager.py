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

    def __init__(self, model, checkpoint, *, compile=True):
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("The native NeRD solver currently requires CUDA.")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != "isaaclab-nerd-v2":
            raise ValueError("Expected a native Isaac Lab NeRD checkpoint.")
        self.dynamics = JointDynamics(saved["dynamics"], model)
        if model_signature(model) != saved["dynamics"]["model"]:
            raise ValueError("NeRD checkpoint physical model differs from the live scene.")
        self.frame_dt = saved["dynamics"]["frame_dt"]
        device = model.device.alias
        self.network = (
            string_to_callable(saved["network_type"])(
                self.dynamics.input_dim, self.dynamics.codec.prediction_shape[0], **saved["network_cfg"]
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
        self.pending_reset = torch.ones(model.world_count, dtype=torch.bool, device=device)
        self.predict = torch.compile(self.network.predict_last) if compile else self.network.predict_last
        # Warm model kernels and compilation outside the host's graph capture. No physical state advances.
        with torch.no_grad(), torch.cuda.stream(wp.stream_to_torch(wp.get_stream(model.device))):
            self.predict(self.history)

    @torch.no_grad()
    def reset(self, state, *, world_mask=None, flags=0):
        with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.model.device))):
            if world_mask is None:
                self.pending_reset.fill_(True)
            else:
                self.pending_reset.logical_or_(wp.to_torch(world_mask)[: self.model.world_count])

    @torch.no_grad()
    def step(self, state_in, state_out, control, contacts, dt):
        if not math.isclose(dt, self.frame_dt, rel_tol=1e-9):
            raise ValueError(f"NeRD frame is {self.frame_dt}s, received {dt}s.")
        with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.model.device))):
            self.dynamics.check_inputs(state_in, control)
            if self.dynamics.descriptor["control"] == "joint_target_q":
                action = wp.to_torch(control.joint_target_q)
            else:
                action = wp.to_torch(control.joint_f)
            state = self.dynamics.codec.read(state_in)
            features = self.dynamics.inputs(
                state, action.reshape(self.model.world_count, -1), self.dynamics.contacts(state_in)
            )
            token = (features - self.mean) / self.scale
            shifted = torch.cat((self.history[:, 1:], token[:, None]), dim=1)
            self.history.copy_(torch.where(self.pending_reset[:, None, None], token[:, None], shifted))
            self.pending_reset.zero_()
            delta = self.predict(self.history) * self.output_scale + self.output_mean
            prediction = self.dynamics.codec.delta_to_state(state, delta)
            torch._assert_async(torch.isfinite(prediction).all(), "NeRD produced a nonfinite state.")
            layout = self.dynamics.codec.layout
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

    def notify_model_changed(self, flags):
        # Fixed-base root-pose resets can notify unchanged joint properties. Verify the same
        # physical contract instead of suppressing every notification or mutating a library method.
        if int(flags) and model_signature(self.model) != self.dynamics.descriptor["model"]:
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
        self._solver = SolverNeRD(model, solver_cfg.checkpoint, compile=solver_cfg.compile)
        if not math.isclose(self._solver_dt, self._solver.frame_dt, rel_tol=1e-9):
            raise ValueError("Set sim.dt to the NeRD checkpoint's control-period duration.")
        self._use_single_state = True
        self._needs_collision_pipeline = False
        self._supports_rigid_body_force_input = False
        self._supports_contact_sensors = False

    def set_decimation(self, decimation):
        if decimation != 1 and self._solver.dynamics.descriptor["control"] == "joint_target_q":
            raise ValueError("Target-conditioned NeRD predicts a complete control period; set decimation=1.")
        super().set_decimation(decimation)

    def _capture_graph(self, capture_target):
        # Warp owns graph capture/replay. Isolate its Torch temporaries from the caller's caching
        # allocator and keep this pool alive as long as the captured physics graph can use them.
        self._torch_pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(self._torch_pool, device=self._device):
            return super()._capture_graph(capture_target)

    def _solver_specific_clear(self):
        self._torch_pool = None
