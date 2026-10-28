# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Native manager terms for reference tracking and physical transition capture."""

from __future__ import annotations

import torch
import warp as wp
from newton import ModelFlags

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import CommandTerm, RecorderTerm
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .kinematics import G1_JOINT_NAMES, G1Kinematics, mat_to_6d, so3_exp
from .motion import BIN_FRAMES, MotionLib
from .tracking import MAX_REWARD_RATE, STATE_DIM, TrackingObjective


class MotionCommand(CommandTerm):
    """Own reference sampling, physical context and the task's analytical objectives."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        if list(self.robot.joint_names) != G1_JOINT_NAMES:
            raise ValueError(f"The tracking asset must preserve the motion joint order: {self.robot.joint_names}")
        self.kin = G1Kinematics(cfg.kinematics_path, self.device)
        self.motion = MotionLib(cfg.motion_dir, self.kin, self.device, cfg.source_fps)
        self.mdp = TrackingObjective(
            self.kin, self.motion, self.robot.data.default_joint_pos.torch[0].clone(), cfg.objective
        )
        self.physics = env.sim.backend_manager
        self.robot_view = self.robot.root_view
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.xi = torch.zeros(self.num_envs, 33, device=self.device)
        self.start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_bin = torch.zeros_like(self.start_frame)
        self._accounted_step = torch.zeros_like(self.start_frame)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.discounted_return = torch.zeros_like(self.episode_return)
        self.bin_return = torch.zeros(self.motion.num_bins, device=self.device)
        self.bin_weight = torch.full_like(self.bin_return, 1.0 / self.motion.num_bins)
        self._command = torch.zeros(self.num_envs, 73 * cfg.goal_frames, device=self.device)
        self.metrics = {
            "return": torch.zeros_like(self.episode_return),
            "length": torch.zeros_like(self.episode_return),
        }

    @property
    def command(self):
        return self._command

    @property
    def frame(self):
        return self.start_frame + self._env.episode_length_buf

    def state(self):
        data = self.robot.data
        pose, velocity = data.root_link_pose_w.torch, data.root_link_vel_w.torch
        return (
            pose[:, :3] - self._env.scene.env_origins,
            matrix_from_quat(pose[:, 3:7]),
            velocity[:, :3],
            velocity[:, 3:],
            data.joint_pos.torch,
            data.joint_vel.torch,
        )

    def _account_reward(self, env_ids):
        steps = self._env.episode_length_buf[env_ids]
        pending = steps > self._accounted_step[env_ids]
        if not hasattr(self._env, "reward_buf"):
            return pending
        reward = torch.where(pending, self._env.reward_buf[env_ids], 0.0)
        self.episode_return[env_ids] += reward
        self.discounted_return[env_ids] += self.cfg.gamma ** (steps.float() - 1) * reward
        self._accounted_step[env_ids] = steps
        return pending

    def reset(self, env_ids=None):
        ids = self.env_ids if env_ids is None else self.env_ids[env_ids]
        pending = self._account_reward(ids)
        # Public reset after command.compute discards a partial episode without updating its bin twice.
        if hasattr(self._env, "reset_terminated") and not self.cfg.evaluation:
            boundary = pending & (self._env.reset_terminated[ids] | self._env.reset_time_outs[ids])
            ended = ids[boundary]
            if len(ended):
                steps = self._env.episode_length_buf[ended].float()
                bootstrap = torch.where(
                    self._env.reset_terminated[ended],
                    0.0,
                    self.cfg.gamma**steps * (self.episode_return[ended] / steps) / (1 - self.cfg.gamma),
                )
                returns = self.discounted_return[ended] + bootstrap
                sums, counts = torch.zeros_like(self.bin_return), torch.zeros_like(self.bin_return)
                sums.index_add_(0, self.start_bin[ended], returns)
                counts.index_add_(0, self.start_bin[ended], torch.ones_like(returns))
                self.bin_return.copy_(
                    torch.where(counts > 0, 0.8 * self.bin_return + 0.2 * sums / counts.clamp_min(1), self.bin_return)
                )
        scale = MAX_REWARD_RATE * self._env.step_dt * self._env.reward_manager.get_term_cfg("tracking").weight
        self.metrics["return"][ids] = self.episode_return[ids] / (
            self._env.episode_length_buf[ids].clamp_min(1) * scale
        )
        self.metrics["length"][ids] = self._env.episode_length_buf[ids].float()
        return super().reset(ids)

    def _update_metrics(self):
        self._account_reward(self.env_ids)

    def _update_command(self):
        self._command.copy_(self.motion.goal(self.frame, self.cfg.goal_frames))

    def _resample_command(self, env_ids):
        ids = self.env_ids[env_ids]
        n, motion = len(ids), self.motion
        if self.cfg.evaluation:
            frames = motion.clip_start[(ids + self.cfg.evaluation_clip_offset) % len(motion.names)]
            bins = motion.frame_bin[frames]
        else:
            maximum_return = (
                MAX_REWARD_RATE
                * self._env.step_dt
                * self._env.reward_manager.get_term_cfg("tracking").weight
                / (1 - self.cfg.gamma)
            )
            soft = torch.softmax(-25.0 * self.bin_return / maximum_return, 0)
            self.bin_weight.mul_(0.999).add_(0.001 * soft)
            probabilities = 0.5 / motion.num_bins + 0.4 * soft + 0.1 * self.bin_weight / self.bin_weight.sum()
            bins = torch.multinomial(probabilities, n, replacement=True)
            start = motion.bin_start[bins]
            end = torch.minimum(start + BIN_FRAMES, motion.clip_end[motion.bin_clip[bins]])
            frames = start + (torch.rand(n, device=self.device) * (end - start).float()).long()

        p, R, v, w, q, qd = (value.clone() for value in motion.ref_state(frames))
        if self.cfg.reset_noise and not self.cfg.evaluation:
            p += torch.empty_like(p).uniform_(-1, 1) * p.new_tensor((0.05, 0.05, 0.01))
            rotation = torch.empty_like(p).uniform_(-1, 1) * p.new_tensor((0.1, 0.1, 0.2))
            R = so3_exp(rotation) @ R
            v += torch.empty_like(v).uniform_(-1, 1) * v.new_tensor((0.5, 0.5, 0.2))
            w += torch.empty_like(w).uniform_(-1, 1) * w.new_tensor((0.52, 0.52, 0.78))
            q += torch.empty_like(q).uniform_(-0.1, 0.1)
        q = q.clamp(self.kin.q_lo, self.kin.q_hi)
        pose = torch.cat((p + self._env.scene.env_origins[ids], quat_from_matrix(R)), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=ids)
        self.robot.write_joint_state_to_sim_index(position=q, velocity=qd, env_ids=ids)
        self.robot.actuators.target_command.set_position_index(value=q, env_ids=ids)
        self.start_frame[ids], self.start_bin[ids] = frames, bins
        self._accounted_step[ids] = 0
        self.episode_return[ids] = 0
        self.discounted_return[ids] = 0
        if not self.cfg.evaluation:
            self.xi[ids, 1:30] = torch.empty(n, 29, device=self.device).uniform_(-0.01, 0.01)
        # The native environment clears episode_length_buf after manager resets.
        self._command[ids] = motion.goal(frames, self.cfg.goal_frames)


class ContextJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.motion = env.command_manager.get_term("motion")

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions.add_(self.motion.xi[:, 1:30])


def randomize_context(env, env_ids):
    command = env.command_manager.get_term("motion")
    if command.cfg.evaluation:
        command.xi[:, 0] = 1.0
    else:
        command.xi[:, 0].uniform_(0.3, 1.2)
        command.xi[:, 30:33] = torch.empty(env.num_envs, 3, device=env.device).uniform_(-1, 1) * command.xi.new_tensor(
            (0.025, 0.05, 0.05)
        )
        default_com = command.robot.data.body_com_pos_b.torch[:, command.torso_id].clone()
        command.robot.set_coms_index(
            coms=(default_com + command.xi[:, 30:33]).unsqueeze(1), body_ids=[command.torso_id]
        )
    model = command.physics.get_model()
    shape = command.robot_view.get_attribute("shape_material_mu", model).shape
    values = command.xi[:, 0].reshape(env.num_envs, 1, 1).expand(shape).contiguous()
    command.robot_view.set_attribute("shape_material_mu", model, wp.from_torch(values))
    command.physics.add_model_change(ModelFlags.SHAPE_PROPERTIES)


def joint_pos_rel(env):
    command = env.command_manager.get_term("motion")
    return command.robot.data.joint_pos.torch - command.mdp.default_q - command.xi[:, 1:30]


def relative_orientation(env):
    command = env.command_manager.get_term("motion")
    R = matrix_from_quat(command.robot.data.root_link_pose_w.torch[:, 3:7])
    return mat_to_6d(R.transpose(-1, -2) @ command.motion.root_R[command.frame])


def tracking_reward(env):
    command = env.command_manager.get_term("motion")
    return command.mdp.reward(command.state(), command.frame, env.action_manager.action, env.action_manager.prev_action)


def tracking_failure(env):
    command = env.command_manager.get_term("motion")
    return command.mdp.termination(command.state(), command.frame)


def clip_end(env):
    command = env.command_manager.get_term("motion")
    return command.frame >= command.motion.clip_end[command.motion.clip_id[command.start_frame]]


class TransitionRecorder(RecorderTerm):
    """Publish actual pre/post tensors until the next step; optionally retain them as episodes.

    An exported episode's end is also a boundary, including explicit resets that happen after a
    transition was retained. Online consumers additionally check adjacent state/step continuity.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.motion = env.command_manager.get_term("motion")
        self.robot = self.motion.robot
        self.latest = {
            "s": torch.empty(self.num_envs, STATE_DIM, device=self.device),
            "s1": torch.empty(self.num_envs, STATE_DIM, device=self.device),
            "a": torch.empty(self.num_envs, 29, device=self.device),
            "a_prev": torch.empty(self.num_envs, 29, device=self.device),
            "xi": torch.empty_like(self.motion.xi),
            "f": torch.empty(self.num_envs, dtype=torch.long, device=self.device),
            "ep_step": torch.empty(self.num_envs, dtype=torch.int32, device=self.device),
            "r": torch.empty(self.num_envs, device=self.device),
            "done": torch.empty(self.num_envs, dtype=torch.bool, device=self.device),
            "term": torch.empty(self.num_envs, dtype=torch.bool, device=self.device),
        }
        self._has_transition = False

    def _capture_state(self, target):
        data = self.robot.data
        target[:, :7].copy_(data.root_link_pose_w.torch)
        target[:, :3].sub_(self._env.scene.env_origins)
        target[:, 7:13].copy_(data.root_link_vel_w.torch)
        target[:, 13:42].copy_(data.joint_pos.torch)
        target[:, 42:71].copy_(data.joint_vel.torch)

    def record_pre_step(self):
        self._capture_state(self.latest["s"])
        self.latest["a"].copy_(self._env.action_manager.action)
        self.latest["a_prev"].copy_(self._env.action_manager.prev_action)
        self.latest["xi"].copy_(self.motion.xi)
        self.latest["f"].copy_(self.motion.frame)
        self.latest["ep_step"].copy_(self._env.episode_length_buf)
        self._has_transition = True
        return None, None

    def record_post_step(self):
        self._capture_state(self.latest["s1"])
        self.latest["r"].copy_(self._env.reward_buf)
        self.latest["done"].copy_(self._env.reset_buf)
        self.latest["term"].copy_(self._env.reset_terminated)
        return ("transition", self.latest) if self.cfg.record_to_episodes else (None, None)

    def record_pre_reset(self, env_ids):
        if self._has_transition:
            self.latest["done"][slice(None) if env_ids is None else env_ids] = True
        return None, None
