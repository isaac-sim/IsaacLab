# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native reference command for model-based motion tracking."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from isaaclab_rl.model_based.math import to_native
from isaaclab_rl.model_based.representations import motion_goal_window

from isaaclab_tasks.core.motion_tracking.mdp.model_based.reference import MotionReference


def _sample_frames(distribution: torch.Tensor, frame_count: int, count: int) -> torch.Tensor:
    """Sample timeline bins without rounding the last frame past the exclusive end."""
    bins = torch.multinomial(distribution, count, replacement=True)
    frames = ((bins + torch.rand(count, device=distribution.device)) / len(distribution) * frame_count).long()
    return frames.clamp_max_(frame_count - 1)


class MotionCommand(CommandTerm):
    """Advance reference frames after rewards; resample only at episode boundaries."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.motion = MotionReference(cfg, self.device)
        if self.robot.joint_names != self.motion.fk.joint_names:
            raise ValueError("Motion joints must match the articulation's native joint order.")
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
        self._sampling_distribution = torch.full((self.bin_count,), 1 / self.bin_count, device=self.device)
        self.encoder_bias = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.friction = torch.ones(self.num_envs, 1, device=self.device)
        self.com = self.robot.data.body_com_pos_b.torch[:, self.robot.body_names.index(cfg.context_body)].clone()
        self.nominal_com = self.com.clone()
        self._command = torch.zeros(self.num_envs, 20 * self.robot.num_joints, device=self.device)
        self.body_indexes = self.robot.find_bodies(self.motion.fk.body_names, preserve_order=True)[0]

    @property
    def command(self):
        return self._command

    def set_sampling_distribution(self, weights: torch.Tensor) -> None:
        """Set the return sampler's distribution over equal timeline bins."""
        if weights.shape != self._sampling_distribution.shape:
            raise ValueError("Sampling weights must contain one value per timeline bin.")
        self._sampling_distribution.copy_(weights)

    def _update_metrics(self):
        pass

    def _update_command(self):
        clip = torch.searchsorted(self.motion.motion_ends, self.time_steps, right=True)
        last = self.motion.motion_ends[clip] - 1
        self.time_steps.copy_(torch.minimum(self.time_steps + (self._env.episode_length_buf > 0).long(), last))
        self._command.copy_(motion_goal_window(self.motion, self.time_steps))

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        count = len(ids)
        if self.cfg.evaluation or self.cfg.sampling_mode == "start":
            frames = self.motion.motion_starts[ids % len(self.motion.names)]
        elif self.cfg.sampling_mode == "uniform":
            frames = torch.randint(self.motion.time_step_total, (count,), device=self.device)
        else:
            frames = _sample_frames(self._sampling_distribution, self.motion.time_step_total, count)
        motion = self.motion
        pos, quat = motion.root_pos_w[frames].clone(), to_native(motion.root_quat_w[frames])
        vel = torch.cat((motion.root_lin_vel_w[frames], motion.root_ang_vel_w[frames]), -1).clone()
        joints = motion.joint_pos[frames].clone()
        if self.cfg.reset_noise and not self.cfg.evaluation:
            delta = torch.empty(count, 6, device=self.device).uniform_(-1, 1)
            delta *= delta.new_tensor((0.05, 0.05, 0.01, 0.1, 0.1, 0.2))
            pos += delta[:, :3]
            quat = quat_mul(quat_from_euler_xyz(*delta[:, 3:].unbind(-1)), quat)
            vel += torch.empty_like(vel).uniform_(-1, 1) * vel.new_tensor((0.5, 0.5, 0.2, 0.52, 0.52, 0.78))
            joints += torch.empty_like(joints).uniform_(-0.1, 0.1)
        limits = self.robot.data.soft_joint_pos_limits.torch[ids]
        joints = joints.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat((pos + self._env.scene.env_origins[ids], quat), -1), env_ids=ids
        )
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
        self.robot.write_joint_state_to_sim_index(position=joints, velocity=motion.joint_vel[frames], env_ids=ids)
        self.robot.actuators.target_command.set_position_index(value=joints, env_ids=ids)
        self.time_steps[ids] = frames
        self._command[ids] = motion_goal_window(motion, frames)


class BiasedJointPositionAction(JointPositionAction):
    """Apply the reference controller's encoder bias to joint position targets."""

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions.sub_(self._env.command_manager.get_term("motion").encoder_bias)
