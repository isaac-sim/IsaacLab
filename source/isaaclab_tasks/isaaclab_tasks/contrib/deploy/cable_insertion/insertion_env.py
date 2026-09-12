# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based insertion env with task-success detection and logging.

Adds success metrics to ``extras["log"]`` for RSL-RL without changing the MDP
observation, action, reward, or termination logic.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms


def _keypoint_offsets_6d(device: torch.device) -> torch.Tensor:
    """Return the 7 unit keypoint offsets used by the keypoint reward."""
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0)


class DisplayportInsertionEnv(ManagerBasedRLEnv):
    """Manager-based RL env that logs insertion success metrics during training.

    The following scalars are added to ``extras["log"]``:

    - ``Metrics/success_rate``: fraction of envs within the success threshold
    - ``Metrics/plug_socket_pos_error_m``: mean mate-point distance (m)
    - ``Metrics/plug_socket_keypoint_dist_m``: mean keypoint distance (m)
    - ``Metrics/terminal_success_rate``: success fraction at episode reset
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._log_success_metrics: bool = bool(getattr(cfg, "log_success_metrics", True))
        self._success_socket_asset: str = getattr(cfg, "success_socket_asset", "dp_socket")
        self._success_plug_asset: str = getattr(cfg, "success_plug_asset", "dp_plug")
        self._success_pos_threshold: float = float(getattr(cfg, "success_pos_threshold", 0.003))
        self._success_keypoint_scale: float = float(getattr(cfg, "success_keypoint_scale", 0.15))

        device = self.device
        self._success_socket_offset = torch.tensor(
            getattr(cfg, "success_socket_offset", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32
        )
        self._success_plug_offset = torch.tensor(
            getattr(cfg, "success_plug_offset", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32
        )
        self._success_plug_goal_rot_inv = torch.tensor(
            getattr(cfg, "success_plug_goal_rot_inv", [0.0, 0.0, 0.0, 1.0]), device=device, dtype=torch.float32
        )

        self._success_identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._success_kp_offsets = _keypoint_offsets_6d(device) * self._success_keypoint_scale

    def _compute_success(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-env success mask, mate-point distance, and keypoint distance."""
        socket = self.scene[self._success_socket_asset]
        plug = self.scene[self._success_plug_asset]

        socket_pos = wp.to_torch(socket.data.root_pos_w)
        socket_quat = wp.to_torch(socket.data.root_quat_w)
        plug_pos = wp.to_torch(plug.data.root_pos_w)
        plug_quat = wp.to_torch(plug.data.root_quat_w)

        n = self.num_envs
        socket_off = self._success_socket_offset.unsqueeze(0).expand(n, -1)
        plug_off = self._success_plug_offset.unsqueeze(0).expand(n, -1)
        plug_goal_rot_inv = self._success_plug_goal_rot_inv.unsqueeze(0).expand(n, -1)

        # Mate reference frames (same construction as keypoint_two_body_error)
        kp_pos_s, kp_quat_s = combine_frame_transforms(socket_pos, socket_quat, socket_off, self._success_identity_quat)
        kp_pos_p, kp_quat_p = combine_frame_transforms(plug_pos, plug_quat, plug_off, plug_goal_rot_inv)

        pos_error = torch.linalg.norm(kp_pos_p - kp_pos_s, dim=-1)

        k = self._success_kp_offsets.shape[0]
        offs_flat = self._success_kp_offsets.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
        ident_flat = self._success_identity_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)

        kp_s = combine_frame_transforms(
            kp_pos_s.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3),
            kp_quat_s.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            offs_flat,
            ident_flat,
        )[0].reshape(n, k, 3)
        kp_p = combine_frame_transforms(
            kp_pos_p.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3),
            kp_quat_p.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            offs_flat,
            ident_flat,
        )[0].reshape(n, k, 3)
        keypoint_dist = torch.linalg.norm(kp_p - kp_s, dim=-1).mean(dim=-1)

        is_success = pos_error < self._success_pos_threshold
        return is_success, pos_error, keypoint_dist

    def step(self, action: torch.Tensor):
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        if getattr(self, "_log_success_metrics", False):
            is_success, pos_error, keypoint_dist = self._compute_success()
            log = self.extras.setdefault("log", {})
            log["Metrics/success_rate"] = is_success.float().mean()
            log["Metrics/plug_socket_pos_error_m"] = pos_error.mean()
            log["Metrics/plug_socket_keypoint_dist_m"] = keypoint_dist.mean()
        return obs_buf, reward_buf, terminated, time_outs, self.extras

    def _reset_idx(self, env_ids):
        terminal_success = None
        if getattr(self, "_log_success_metrics", False):
            is_success, _, _ = self._compute_success()
            terminal_success = is_success[env_ids].float().mean()

        super()._reset_idx(env_ids)

        if terminal_success is not None:
            self.extras["log"]["Metrics/terminal_success_rate"] = terminal_success
