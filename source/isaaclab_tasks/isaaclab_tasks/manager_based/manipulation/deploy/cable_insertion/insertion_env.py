# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based insertion env with task-success detection and logging.

This is a thin :class:`~isaaclab.envs.ManagerBasedRLEnv` subclass that adds a
GB300-style success metric (does the held plug sit at the socket's mated pose?)
and logs the success rate to ``extras["log"]`` so RSL-RL surfaces it in
TensorBoard / Weights & Biases during training. It does **not** change the
observation, action, reward, or termination logic, so training dynamics are
identical to the stock ``ManagerBasedRLEnv``.

Success is measured from the same keypoint frames the keypoint-tracking reward
uses: the socket's mate reference point (root + ``socket_offset``) and the
plug's mate reference point (root + ``plug_offset`` with the goal rotation
offset applied). At the verified seated pose these frames coincide, so a small
mate-point distance means the plug is inserted. We deliberately reuse the exact
``combine_frame_transforms`` call pattern and the same offset constants as the
reward term to stay agnostic to the (x, y, z, w) quaternion convention of the
Newton/warp backend.
"""

from __future__ import annotations

import torch

import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms


def _keypoint_offsets_6d(device: torch.device) -> torch.Tensor:
    """Return the 7 unit keypoint offsets (center + ±x/±y/±z) used by the reward.

    Mirrors ``deploy.mdp.rewards._get_keypoint_offsets_full_6d(add_cube_center_kp=True)``
    so the logged keypoint distance matches the training reward's keypoint distance.
    """
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0)  # (7, 3)


class DisplayportInsertionEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` that additionally computes and logs a task-success rate.

    The following scalars are added to ``extras["log"]`` (RSL-RL logs them):

    - ``Metrics/success_rate``: instantaneous fraction of envs whose plug mate
      point is within ``success_pos_threshold`` of the socket mate point (logged
      every step).
    - ``Metrics/plug_socket_pos_error_m``: mean mate-point distance (m).
    - ``Metrics/plug_socket_keypoint_dist_m``: mean keypoint distance (m), the
      same quantity the keypoint reward minimizes.
    - ``Metrics/terminal_success_rate``: success fraction evaluated on envs at the
      instant they reset (episode-end success), logged on reset steps.

    All parameters are read from the env cfg (see ``DisplayportInsertionEnvCfg``)
    with safe fallbacks, so this class also works with cfgs that do not define
    the success fields.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Read (optional) success-logging config with safe defaults. These are set
        # after ``super().__init__`` so the ``_reset_idx`` override that runs during
        # construction skips logging (guarded via ``getattr``).
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

        # Pre-allocated constants for the keypoint computation.
        self._success_identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._success_kp_offsets = _keypoint_offsets_6d(device) * self._success_keypoint_scale  # (7, 3)

    def _compute_success(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-env success mask, mate-point distance, and keypoint distance.

        Returns:
            Tuple ``(is_success, pos_error, keypoint_dist)`` each of shape ``(num_envs,)``.
        """
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

        # Mate reference frames (identical construction to keypoint_two_body_error).
        kp_pos_s, kp_quat_s = combine_frame_transforms(socket_pos, socket_quat, socket_off, self._success_identity_quat)
        kp_pos_p, kp_quat_p = combine_frame_transforms(plug_pos, plug_quat, plug_off, plug_goal_rot_inv)

        pos_error = torch.linalg.norm(kp_pos_p - kp_pos_s, dim=-1)

        # Mean keypoint distance over the 7 spread keypoints (captures orientation too).
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
        # Evaluate success on the pre-reset state of the resetting envs so we get a
        # clean per-episode ("terminal") success rate. Guarded with getattr because
        # the parent constructor calls _reset_idx before our attributes are set.
        terminal_success = None
        if getattr(self, "_log_success_metrics", False):
            is_success, _, _ = self._compute_success()
            terminal_success = is_success[env_ids].float().mean()

        super()._reset_idx(env_ids)

        # super()._reset_idx() reset extras["log"] and repopulated it with the
        # Episode_* stats; append our terminal metric so it survives to step()'s return.
        if terminal_success is not None:
            self.extras["log"]["Metrics/terminal_success_rate"] = terminal_success
