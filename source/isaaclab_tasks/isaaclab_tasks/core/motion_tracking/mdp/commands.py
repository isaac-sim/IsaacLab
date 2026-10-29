# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Reference motion commands and failure-adaptive phase sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from ..data.motion_loader import MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionCommand(CommandTerm):
    """Command term for the motion tracking task."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indices = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion = MotionLoader(
            self.cfg.motion_file, body_names=self.cfg.body_names, joint_names=self.robot.joint_names, device=self.device
        )
        if not math.isclose(self.motion.fps * env.step_dt, 1.0, rel_tol=1e-5):
            raise ValueError("Motion FPS must match the policy rate; resample the reference before training.")
        if cfg.body_names[0] != self.robot.body_names[0]:
            raise ValueError("The first tracked body must be the articulation root.")
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 3] = 1.0
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        self.metrics = {
            "error_anchor_pos": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_rot": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "error_body_pos": torch.zeros(self.num_envs, device=self.device),
            "error_body_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "error_body_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "error_body_rot": torch.zeros(self.num_envs, device=self.device),
            "error_joint_pos": torch.zeros(self.num_envs, device=self.device),
            "error_joint_vel": torch.zeros(self.num_envs, device=self.device),
            "sampling_entropy": torch.zeros(self.num_envs, device=self.device),
            "sampling_top1_prob": torch.zeros(self.num_envs, device=self.device),
            "sampling_top1_bin": torch.zeros(self.num_envs, device=self.device),
        }

    def __str__(self) -> str:
        msg = "MotionCommand:\n"
        msg += f"\tAsset name: {self.cfg.asset_name}\n"
        msg += f"\tMotion file: {self.cfg.motion_file}\n"
        msg += f"\tAnchor body name: {self.cfg.anchor_body_name}\n"
        msg += f"\tBody names: {self.cfg.body_names}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, 2 * num_joints)."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """The desired joint positions. Shape is (num_envs, num_joints)."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """The desired joint velocities. Shape is (num_envs, num_joints)."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """The desired body positions in the world frame. Shape is (num_envs, num_bodies, 3)."""
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """The desired body quaternions in the world frame. Shape is (num_envs, num_bodies, 4)."""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """The desired body linear velocities in the world frame. Shape is (num_envs, num_bodies, 3)."""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """The desired body angular velocities in the world frame. Shape is (num_envs, num_bodies, 3)."""
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """The desired anchor position in the world frame. Shape is (num_envs, 3)."""
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """The desired anchor quaternion in the world frame. Shape is (num_envs, 4)."""
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """The desired anchor linear velocity in the world frame. Shape is (num_envs, 3)."""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """The desired anchor angular velocity in the world frame. Shape is (num_envs, 3)."""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """The current body positions of the robot in the world frame. Shape is (num_envs, num_bodies, 3)."""
        return self.robot.data.body_pos_w.torch[:, self.body_indices]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """The current body quaternions of the robot in the world frame. Shape is (num_envs, num_bodies, 4)."""
        return self.robot.data.body_quat_w.torch[:, self.body_indices]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """The current body linear velocities of the robot in the world frame. Shape is (num_envs, num_bodies, 3)."""
        return self.robot.data.body_lin_vel_w.torch[:, self.body_indices]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """The current body angular velocities of the robot in the world frame. Shape is (num_envs, num_bodies, 3)."""
        return self.robot.data.body_ang_vel_w.torch[:, self.body_indices]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """The current anchor position of the robot in the world frame. Shape is (num_envs, 3)."""
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """The current anchor quaternion of the robot in the world frame. Shape is (num_envs, 4)."""
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """The current anchor linear velocity of the robot in the world frame. Shape is (num_envs, 3)."""
        return self.robot.data.body_lin_vel_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """The current anchor angular velocity of the robot in the world frame. Shape is (num_envs, 3)."""
        return self.robot.data.body_ang_vel_w.torch[:, self.robot_anchor_body_index]

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot.data.joint_pos.torch, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot.data.joint_vel.torch, dim=-1)

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return
        self._adaptive_sampling(env_ids)
        frames = self.time_steps[env_ids]
        root_pos = self.motion.body_pos_w[frames, 0] + self._env.scene.env_origins[env_ids]
        root_ori = self.motion.body_quat_w[frames, 0]
        root_lin_vel = self.motion.body_lin_vel_w[frames, 0]
        root_ang_vel = self.motion.body_ang_vel_w[frames, 0]

        ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=self.device,
        )
        offsets = sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=self.device)
        root_pos += offsets[:, :3]
        root_ori = quat_mul(quat_from_euler_xyz(offsets[:, 3], offsets[:, 4], offsets[:, 5]), root_ori)
        ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=self.device,
        )
        offsets = sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=self.device)
        root_lin_vel += offsets[:, :3]
        root_ang_vel += offsets[:, 3:]
        joint_pos = self.motion.joint_pos[frames]
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, self.motion.joint_vel[frames], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )
        self._update_reference_alignment()

    def _update_command(self) -> None:
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)
        self._update_reference_alignment()
        self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
        self._current_bin_failed.zero_()

    def _update_reference_alignment(self) -> None:
        # Match the robot's horizontal position and yaw, retaining reference height and tilt.
        delta_pos = self.robot_anchor_pos_w.clone()
        delta_pos[:, 2] = self.anchor_pos_w[:, 2]
        delta_quat = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        delta_quat = delta_quat[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        self.body_quat_relative_w = quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = delta_pos[:, None, :] + quat_apply(
            delta_quat, self.body_pos_w - self.anchor_pos_w[:, None, :]
        )

    def _adaptive_sampling(self, env_ids: Sequence[int] | slice) -> None:
        """Adaptive sampling to compute the starting index of the motion."""
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                self.time_steps * self.bin_count // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed += torch.bincount(fail_bins, minlength=self.bin_count)
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        sampled_bins = torch.multinomial(sampling_probabilities, num_envs, replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (num_envs,), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else torch.zeros_like(H)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Named reference motion with failure-adaptive phase sampling."""

    class_type: type = MotionCommand
    asset_name: str = MISSING
    """Articulation to control."""
    motion_file: str = MISSING
    """NPZ path; see the data loader for the array format."""
    anchor_body_name: str = MISSING
    """Body defining the reference alignment frame."""
    body_names: list[str] = MISSING
    """Tracked bodies in order, starting with the articulation root."""
    pose_range: dict[str, tuple[float, float]] = {}
    """Reset offsets: x/y/z [m] and roll/pitch/yaw [rad]."""
    velocity_range: dict[str, tuple[float, float]] = {}
    """Reset velocity offsets: x/y/z [m/s] and roll/pitch/yaw [rad/s]."""
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    """Joint reset perturbation [rad]."""
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
