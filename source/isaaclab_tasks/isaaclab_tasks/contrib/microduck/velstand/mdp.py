# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current merged upstream VelStand terms, isolated from every legacy task.

Source: pollen-robotics/microduck_rl@bae39f5900b318f7e87fc485c30a31213cfa14eb.
Quaternions use native Isaac Lab xyzw; reset heights are relative to each terrain origin.
"""

from __future__ import annotations

import math
from types import MethodType

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import resolve_matching_names_values

import isaaclab_tasks.contrib.microduck.mdp as shared


def _finite_reward(value):
    """Match the source reward guard locally while the invalid-state termination resets physics."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_reward_compute(manager, dt):
    """Native aggregation with the source MJLab per-term finite guard before accumulation.

    Keep native per-second step buffers and dt-scaled totals/history. Guarding only
    the total would erase other terms and previous episode sums on a bad sample.
    """
    for index, cfg in enumerate(manager._term_cfgs):
        step_reward = manager._step_reward[:, index]
        if cfg.weight == 0.0:
            step_reward.zero_()
            continue
        value = cfg.func(manager._env, **cfg.params)
        if isinstance(value, torch.Tensor) and value.shape == step_reward.shape:
            torch.mul(value, cfg.weight, out=step_reward)
        else:
            step_reward[:] = value * cfg.weight
        torch.nan_to_num_(step_reward, nan=0.0, posinf=0.0, neginf=0.0)
    torch.sum(manager._step_reward, dim=1, out=manager._reward_buf).mul_(dt)
    manager._episode_sum_buf.add_(manager._step_reward, alpha=dt)
    return manager._reward_buf


def install_finite_reward_guard(env, env_ids):
    """Apply source reward safety only to this VelStand manager, leaving term/reset APIs intact."""
    env.reward_manager.compute = MethodType(_finite_reward_compute, env.reward_manager)


class StandingGatedPoseCommand(shared.ZeroablePoseDeltaCommand):
    """Hold sampled commands privately and expose exact zero outside the standing bucket."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros_like(self._command)

    def _gate(self):
        standing = self._env.command_manager.get_term(self.cfg.twist_command_name).is_standing_env
        self._command[:] = self._raw * standing.unsqueeze(1)

    def _update_command(self):
        self._gate()

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        super()._resample_command(env_ids)
        raw = self._command[env_ids].clone()
        for axis, probability in enumerate(self.cfg.axis_zero_prob):
            if probability > 0:
                raw[torch.rand(len(env_ids), device=self.device) < probability, axis] = 0
        self._raw[env_ids] = raw
        self._gate()


@configclass
class StandingGatedPoseCommandCfg(shared.ZeroablePoseDeltaCommandCfg):
    """Standing-only pose command, with a tilt-only sub-bucket for the stand teacher."""

    class_type = StandingGatedPoseCommand
    twist_command_name: str = "base_velocity"
    axis_zero_prob: tuple[float, ...] = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0)


def _fallen_scale(env, asset_cfg, fallen_scale, gate_tilt_above_deg):
    q = env.scene[asset_cfg.name].data.root_link_quat_w.torch
    cos_tilt = 1 - 2 * (q[:, 0].square() + q[:, 1].square())
    return torch.where(cos_tilt < math.cos(math.radians(gate_tilt_above_deg)), fallen_scale, 1.0)


def action_rate_l2_fallen_scaled(env, fallen_scale=0.1, gate_tilt_above_deg=40.0, asset_cfg=SceneEntityCfg("robot")):
    """Relax the walk's final smoothness cost only while genuinely toppled."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return _finite_reward(delta.square().sum(1) * _fallen_scale(env, asset_cfg, fallen_scale, gate_tilt_above_deg))


class joint_torque_rate_l2_fallen_scaled(ManagerTermBase):
    """Source torque difference, using public native actuator effort.

    Source intentionally retains its previous torque across episode resets. The first ever
    call returns zero; unlike the old port this adapter does not clear it on every reset.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = None

    def __call__(self, env, asset_cfg, fallen_scale=0.1, gate_tilt_above_deg=40.0):
        torque = env.scene[asset_cfg.name].actuators.applied_effort.torch[:, asset_cfg.joint_ids]
        valid = torch.isfinite(torque)
        previous = torch.where(valid, torque, 0.0) if self._previous is None else self._previous
        cost = (torque - previous).square().sum(1)
        # A diverged step is scored zero, and cannot contaminate the next episode's
        # retained torque history. Valid torque retains the source reset semantics.
        self._previous = torch.where(valid, torque, previous).detach().clone()
        return _finite_reward(cost * _fallen_scale(env, asset_cfg, fallen_scale, gate_tilt_above_deg))


def servo_stall_penalty(env, asset_cfg, torque_thresh=0.4, vel_thresh=0.5):
    """Count driven servos producing high actual effort against little motion."""
    asset = env.scene[asset_cfg.name]
    torque = torch.nan_to_num(asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids])
    velocity = torch.nan_to_num(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
    return ((torque.abs() > torque_thresh) & (velocity.abs() < vel_thresh)).float().sum(1)


class servo_acc_spike_penalty(ManagerTermBase):
    """Control-rate velocity differences; first frame of each new episode costs zero."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = None

    def __call__(self, env, asset_cfg, acc_thresh=300.0):
        velocity = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
        valid = torch.isfinite(velocity)
        previous = torch.where(valid, velocity, 0.0) if self._previous is None else self._previous
        acceleration = (velocity - previous) / env.step_dt
        acceleration[env.episode_length_buf <= 1] = 0
        self._previous = torch.where(valid, velocity, previous).detach().clone()
        return _finite_reward((acceleration.abs() - acc_thresh).clamp_min(0).sum(1))


def body_pose_tracking_standing(
    env,
    command_name,
    nominal_height,
    z_std,
    angle_std,
    twist_command_name="base_velocity",
    asset_cfg=SceneEntityCfg("robot"),
):
    """Source z/roll/pitch weighted mean; zero while the twist bucket is moving."""
    value = shared.body_pose_tracking_6d(
        env, command_name, nominal_height, 0.02, z_std, angle_std, asset_cfg, (0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
    )
    standing = env.command_manager.get_term(twist_command_name).is_standing_env
    return _finite_reward(value * standing)


def upright_body_cmd_relative(env, std, command_name, asset_cfg=SceneEntityCfg("robot")):
    """Measure projected gravity against commanded roll/pitch, preserving zero-command walking."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w.torch
    if asset_cfg.body_names is not None:
        quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids].squeeze(1)
    gravity = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    measured = math_utils.quat_apply_inverse(quat, gravity)
    cmd = env.command_manager.get_command(command_name)
    target = torch.stack((torch.sin(cmd[:, 4]), -torch.sin(cmd[:, 3]) * torch.cos(cmd[:, 4])), dim=1)
    return _finite_reward(torch.exp(-(measured[:, :2] - target).square().sum(1) / std**2))


class variable_posture_body_relaxed(shared.pose_mode_switch):
    """Add a fourth posture tolerance only for active standing body commands."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ac = cfg.params["asset_cfg"]
        names = [env.scene[ac.name].joint_names[j] for j in ac.joint_ids]
        _, _, values = resolve_matching_names_values(cfg.params["std_body"], names)
        self._std_body = torch.tensor(values, device=env.device).unsqueeze(0)

    def __call__(
        self,
        env,
        command_name,
        std_standing,
        std_walking,
        std_running,
        walking_threshold,
        running_threshold,
        asset_cfg,
        std_body,
        body_command_name="body_pose",
    ):
        cmd = env.command_manager.get_command(command_name)
        speed = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        standing = speed < walking_threshold
        walking = (speed >= walking_threshold) & (speed < running_threshold)
        running = speed >= running_threshold
        body = env.command_manager.get_command(body_command_name)[:, [2, 3, 4]]
        active = standing & (body.abs().amax(1) > 0)
        std = (
            self._std["std_standing"] * (standing & ~active).unsqueeze(1)
            + self._std_body * active.unsqueeze(1)
            + self._std["std_walking"] * walking.unsqueeze(1)
            + self._std["std_running"] * running.unsqueeze(1)
        )
        asset = env.scene[asset_cfg.name]
        error = (
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        )
        return _finite_reward(torch.exp(-(error.square() / std.square()).mean(1)))


def track_yaw_rate_fine(env, std, command_name, asset_cfg=SceneEntityCfg("robot")):
    """Add the source yaw-only term alongside the unchanged stock angular tracking term."""
    cmd = env.command_manager.get_command(command_name)
    velocity = env.scene[asset_cfg.name].data.root_link_ang_vel_b.torch[:, 2]
    return _finite_reward(torch.exp(-(cmd[:, 2] - velocity).square() / std**2))


def reset_velstand_state(
    env,
    env_ids,
    asset_cfg,
    prone_prob=0.0,
    face_down_prob=1.0,
    crouch_prob=0.0,
    side_prob=0.5,
    joint_random_prob=0.5,
    joint_range_frac=0.9,
    so3_prob=0.25,
    sit_prob=0.05,
    prone_z_range=(0.05, 0.09),
    joint_random_extra_z=0.06,
    so3_extra_z=0.04,
):
    """Exclusive prone/crouch/sit/untouched mixture, including side and uniform-SO(3) starts.

    The SO(3) quarter always has random servo joints. OR with the independent half-joint
    bucket gives a 62.5% random-joint marginal within prone, exactly as the merged source.
    Only selected buckets are rewritten; ordinary standing keeps the inherited reset verbatim.
    """
    if min(prone_prob, crouch_prob, sit_prob) < 0 or prone_prob + crouch_prob + sit_prob > 1:
        raise ValueError("VelStand reset probabilities must form exclusive slices within [0,1]")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if len(env_ids) == 0:
        return
    draw = torch.rand(len(env_ids), device=env.device)
    prone = env_ids[draw < prone_prob]
    crouch = env_ids[(draw >= prone_prob) & (draw < prone_prob + crouch_prob)]
    sitting = env_ids[(draw >= prone_prob + crouch_prob) & (draw < prone_prob + crouch_prob + sit_prob)]
    asset = env.scene[asset_cfg.name]
    if len(prone):
        n = len(prone)
        yaw = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
        side = torch.rand(n, device=env.device) < side_prob
        down = torch.rand(n, device=env.device) < face_down_prob
        sign = torch.where(torch.rand(n, device=env.device) < 0.5, 1.0, -1.0)
        pitch = torch.where(side, 0.0, torch.where(down, math.pi / 2, -math.pi / 2))
        roll = torch.where(side, sign * math.pi / 2, 0.0)
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        so3 = torch.rand(n, device=env.device) < so3_prob
        quat[so3] = torch.nn.functional.normalize(torch.randn(int(so3.sum()), 4, device=env.device), dim=1)
        random_joints = (torch.rand(n, device=env.device) < joint_random_prob) | so3
        height = torch.empty(n, device=env.device).uniform_(*prone_z_range)
        height += random_joints * joint_random_extra_z + so3 * so3_extra_z
        pose = torch.cat((asset.data.root_link_pos_w.torch[prone], quat), dim=1)
        pose[:, 2] = env.scene.env_origins[prone, 2] + height
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=prone)
        asset.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=prone)
        ids = prone[random_joints]
        if len(ids):
            positions = asset.data.default_joint_pos.torch[ids].clone()
            limits = asset.data.joint_pos_limits.torch[ids][:, asset_cfg.joint_ids]
            center = limits.mean(-1)
            half = (limits[..., 1] - limits[..., 0]) * (0.5 * joint_range_frac)
            positions[:, asset_cfg.joint_ids] = center + (2 * torch.rand_like(center) - 1) * half
            asset.write_joint_position_to_sim_index(position=positions, env_ids=ids)
            asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions), env_ids=ids)
    common = dict(face_down_prob=0.0, face_up_prob=0.0, sitting_prob=0.0, standing_prob=0.0, asset_cfg=asset_cfg)
    if len(crouch):
        shared.reset_ground_state(
            env,
            crouch,
            **common,
            crouch_prob=1.0,
            crouch_z_range=(0.06, 0.115),
            crouch_depth_range=(0.35, 1.0),
            crouch_pitch_max=math.radians(55),
            crouch_joint_noise=0.12,
            crouch_joint_pos={
                "left_hip_pitch": -1.15,
                "left_knee": 1.25,
                "left_ankle": 1.05,
                "right_hip_pitch": 1.15,
                "right_knee": -1.25,
                "right_ankle": -1.05,
            },
        )
    if len(sitting):
        common["sitting_prob"] = 1.0
        shared.reset_ground_state(
            env,
            sitting,
            **common,
            sitting_z_range=(0.06, 0.09),
            sitting_joint_noise_std=0.15,
            sitting_tilt_max=math.radians(20),
            sitting_joint_pos={
                "left_hip_roll": 0.0,
                "left_hip_pitch": -0.4079,
                "left_knee": 1.35,
                "left_ankle": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": 0.4079,
                "right_knee": -1.35,
                "right_ankle": 0.0,
            },
        )
