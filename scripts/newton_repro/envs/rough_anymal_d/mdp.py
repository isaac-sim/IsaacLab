# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP for ANYmal-D NaN repro using Newton raw API."""

import math
import os

import numpy as np
import torch
import warp as wp

import newton

NUM_DOFS = 12
ACTION_SCALE = 0.5
# Newton joint order: LF_HAA, LF_HFE, LF_KFE, LH_HAA, LH_HFE, LH_KFE,
#                     RF_HAA, RF_HFE, RF_KFE, RH_HAA, RH_HFE, RH_KFE
ANYMAL_DEFAULT_JOINTS = [0.0, 0.4, -0.8, 0.0, -0.4, 0.8,
                          0.0, 0.4, -0.8, 0.0, -0.4, 0.8]
LSTM_URL = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"


def _quat_rotate_inverse(q, v):
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


class AnymalMDP:
    """ANYmal-D MDP: observations, actions, termination, reset, LSTM actuator.

    Usage::

        mdp = AnymalMDP(...)
        obs = mdp.get_observations()
        while running:
            actions = policy(obs)
            mdp.set_actions(actions)
            for _ in range(decimation):
                mdp.apply_lstm_torques()
                sim_step()
            obs, terminated, truncated = mdp.forward()
            reset_ids = (terminated | truncated).nonzero(...)
            if len(reset_ids) > 0:
                mdp.reset(reset_ids)
    """

    def __init__(self, model, state, control, env_origins,
                 num_envs, jc_per, jd_per, physics_dt, episode_length_s,
                 decimation, device):
        self.model = model
        self.state = state
        self.control = control
        self.num_envs = num_envs
        self.jc_per = jc_per
        self.jd_per = jd_per
        self.decimation = decimation
        self.device = device

        step_dt = physics_dt * decimation
        self.max_episode_length = int(math.ceil(episode_length_s / step_dt))
        self.command_change_steps = int(10.0 / step_dt)

        self.default_joints = torch.tensor(
            ANYMAL_DEFAULT_JOINTS, device=device, dtype=torch.float32
        ).unsqueeze(0)
        self.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        self.env_origins = env_origins
        terrain_file = os.path.join(os.path.dirname(__file__), "terrain_origins.npy")
        raw = np.load(terrain_file)
        self.terrain_origins = torch.tensor(raw.reshape(-1, 3), device=device, dtype=torch.float32)


        # LSTM actuator
        self.lstm = torch.hub.load_state_dict_from_url(
            LSTM_URL, map_location=device, check_hash=False,
            file_name="anydrive_3_lstm_jit.pt",
        )
        self.lstm_h = torch.zeros(2, num_envs * NUM_DOFS, 8, device=device)
        self.lstm_c = torch.zeros(2, num_envs * NUM_DOFS, 8, device=device)

        # Buffers
        self.last_action = torch.zeros(num_envs, NUM_DOFS, device=device)
        self.commands = torch.zeros(num_envs, 3, device=device)
        self.commands[:, 0] = 1.0
        self.episode_length = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.targets = self.default_joints.expand(num_envs, -1).clone()
        self.timestep = 0

        # Match training's Articulation._resolve_actuator_values() for explicit
        # (LSTM) actuators: zero stiffness/damping, set armature, effort limit,
        # velocity limit, and friction on the Newton model arrays.
        wp.to_torch(model.joint_armature).fill_(0.01)
        wp.to_torch(model.joint_target_ke).fill_(0.0)
        wp.to_torch(model.joint_target_kd).fill_(0.0)
        wp.to_torch(model.joint_friction).fill_(0.0)
        wp.to_torch(model.joint_effort_limit).fill_(120.0)

        # Set default standing pose + initial reset
        jq = wp.to_torch(state.joint_q).reshape(num_envs, jc_per)
        jq[:, 7:] = self.default_joints
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        all_ids = torch.arange(num_envs, dtype=torch.int32, device=device)
        self.reset(all_ids)

    def get_observations(self):
        """Compute observations from current state."""
        jq = wp.to_torch(self.state.joint_q).reshape(self.num_envs, self.jc_per)
        jqd = wp.to_torch(self.state.joint_qd).reshape(self.num_envs, self.jd_per)
        root_quat = jq[:, 3:7]
        base_lin_vel = _quat_rotate_inverse(root_quat, jqd[:, :3])
        base_ang_vel = _quat_rotate_inverse(root_quat, jqd[:, 3:6])
        projected_gravity = _quat_rotate_inverse(root_quat, self.gravity_vec.expand(self.num_envs, -1))

        obs = torch.cat([base_lin_vel, base_ang_vel, projected_gravity,
                         self.commands, jq[:, 7:] - self.default_joints,
                         jqd[:, 6:], self.last_action], dim=1)
        return obs

    def set_actions(self, actions):
        """Store actions and compute joint position targets."""
        self.last_action = actions.clone()
        self.targets = self.default_joints + ACTION_SCALE * actions

    def apply_lstm_torques(self):
        """Compute LSTM actuator torques and write to control.joint_f."""
        jq = wp.to_torch(self.state.joint_q).reshape(self.num_envs, self.jc_per)
        jqd = wp.to_torch(self.state.joint_qd).reshape(self.num_envs, self.jd_per)
        sea_in = torch.stack([
            (self.targets - jq[:, 7:]).flatten(),
            jqd[:, 6:].flatten(),
        ], dim=-1).unsqueeze(1)
        torques, (self.lstm_h[:], self.lstm_c[:]) = self.lstm(sea_in, (self.lstm_h, self.lstm_c))
        torques = torques.reshape(self.num_envs, NUM_DOFS).clamp(-80.0, 80.0)

        jf = wp.to_torch(self.control.joint_f).reshape(self.num_envs, self.jd_per)
        jf[:, :6] = 0.0
        jf[:, 6:] = torques

    def forward(self):
        """Compute observations, termination, and resample commands after physics step.

        Returns:
            Tuple of (obs, terminated, truncated).
        """
        obs = self.get_observations()

        self.episode_length += 1
        jq = wp.to_torch(self.state.joint_q).reshape(self.num_envs, self.jc_per)
        root_quat = jq[:, 3:7]
        projected_gravity = _quat_rotate_inverse(root_quat, self.gravity_vec.expand(self.num_envs, -1))
        jqd = wp.to_torch(self.state.joint_qd).reshape(self.num_envs, self.jd_per)
        lin_vel = jqd[:, :3]
        ang_vel = jqd[:, 3:6]
        terminated = (
            (projected_gravity[:, 2] > -0.1736)
            | (lin_vel.norm(dim=1) > 20.0)
            | (ang_vel.norm(dim=1) > 100.0)
        )
        truncated = self.episode_length >= self.max_episode_length

        if self.timestep % self.command_change_steps == 0:
            self.commands[:, 0].uniform_(-1.0, 1.0)
            self.commands[:, 1].uniform_(-0.5, 0.5)
            self.commands[:, 2].uniform_(-0.5, 0.5)


        self.timestep += 1
        return obs, terminated, truncated

    def reset(self, reset_ids):
        """Reset specified environments to randomized standing pose."""
        n = len(reset_ids)
        if n == 0:
            return

        jq = wp.to_torch(self.state.joint_q).reshape(self.num_envs, self.jc_per)
        jqd = wp.to_torch(self.state.joint_qd).reshape(self.num_envs, self.jd_per)

        # Pick random terrain origins for reset position
        rand_idx = torch.randint(0, len(self.terrain_origins), (n,), device=self.device)
        origins = self.terrain_origins[rand_idx]
        jq[reset_ids, 0] = origins[:, 0] + torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
        jq[reset_ids, 1] = origins[:, 1] + torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
        jq[reset_ids, 2] = origins[:, 2] + 0.6
        yaw = torch.empty(n, device=self.device).uniform_(-3.14, 3.14)
        jq[reset_ids, 3] = 0.0
        jq[reset_ids, 4] = 0.0
        jq[reset_ids, 5] = torch.sin(yaw * 0.5)
        jq[reset_ids, 6] = torch.cos(yaw * 0.5)
        jq[reset_ids, 7:] = self.default_joints * torch.empty(n, NUM_DOFS, device=self.device).uniform_(0.5, 1.5)

        jqd[reset_ids] = 0.0

        self.last_action[reset_ids] = 0.0
        offsets = (reset_ids.unsqueeze(1) * NUM_DOFS +
                   torch.arange(NUM_DOFS, device=self.device).unsqueeze(0)).flatten()
        self.lstm_h[:, offsets] = 0.0
        self.lstm_c[:, offsets] = 0.0

        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.episode_length[reset_ids] = 0
