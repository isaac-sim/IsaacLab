# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native reference command for model-based motion tracking."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from isaaclab_rl.model_based.kinematics import to_native
from isaaclab_rl.model_based.representations import motion_goal_window

from isaaclab_tasks.core.motion_tracking.mdp.model_based.reference import MotionReference

from .commands_cfg import ReturnSamplerCfg


# ReturnSampler and ReturnSamplerCfg are derived from HybridRobotics/SuperMimic
# (Apache-2.0), revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
class ReturnSampler:
    def __init__(
        self,
        cfg: ReturnSamplerCfg,
        bin_count: int,
        frame_count: int,
        num_envs: int,
        max_step_reward: float,
        device: str = "cpu",
    ) -> None:
        assert 0.0 < cfg.gamma < 1.0, f"gamma must be in (0, 1), got {cfg.gamma}"
        assert cfg.explore + cfg.keep < 1.0, "explore + keep must leave room for focus"
        self.cfg = cfg
        self.bin_count = bin_count
        self.frame_count = frame_count
        self.v_max = max_step_reward / (1.0 - cfg.gamma)
        # A bin reads v_max until it is first measured.
        self.value = torch.full((bin_count,), self.v_max, device=device)
        self.count = torch.zeros(bin_count, device=device)
        # Summed over the whole run, so float64 keeps the small per-bin terms exact.
        self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
        shape = (num_envs, cfg.horizon)
        self._frames = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._step = 0

    def _bins(self, frames: torch.Tensor) -> torch.Tensor:
        """Frame -> bin, as the motion command bins its adaptive-sampling failures."""
        return frames * self.bin_count // self.frame_count

    def record(
        self,
        frames: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> torch.Tensor | None:
        """Buffer one env step: each env's frame before the step, its reward, and how
        its episode ended, if it did (a fall terminates; reaching the clip's last frame
        and a time-out truncate). Every `horizon` steps, update V and return the new
        reset distribution; otherwise None."""
        col = self._step
        self._frames[:, col] = frames
        self._rewards[:, col] = rewards
        self._terminated[:, col] = terminated
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._step += 1
        if self._step < self.cfg.horizon:
            return None
        self._step = 0
        self._update()
        return self._distribution()

    def _dones(self) -> torch.Tensor:
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self) -> torch.Tensor:
        """Discounted return of every buffered step, computed backwards like
        gae_returns with lambda 1. Where an episode ends, and at the buffer's last step,
        the return continues from the value of the state after the step: none after a
        fall; the mean of V after the clip's last frame, whose successor has no bin on
        the timeline; V at the next frame's bin after a time-out or the buffer's end."""
        gamma = self.cfg.gamma
        rewards = self._rewards
        next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
        next_value = torch.where(
            self._terminated,
            0.0,
            torch.where(self._clip_end, self.value.mean(), self.value[self._bins(next_frames)]),
        )
        dones = self._dones()
        returns = torch.empty_like(rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.cfg.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self) -> None:
        """Fold the buffer into V. Each env's visits to one bin within one episode give
        one sample, their mean return; a bin's earlier samples count at most `memory`
        times."""
        returns = self._returns()
        dones = self._dones()
        num_envs, horizon = dones.shape
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
        key = (envs * horizon + episode) * self.bin_count + self._bins(self._frames)
        keys, inverse, visits = torch.unique(key.reshape(-1), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=returns.device)
        samples.scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.bin_count
        new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        self.count = self.count.clamp(max=self.cfg.memory) + new
        self.value = torch.where(new > 0, self.value + (total - new * self.value) / self.count, self.value)

    def _distribution(self) -> torch.Tensor:
        cfg = self.cfg
        focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        explore = cfg.explore / self.bin_count
        return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep


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
