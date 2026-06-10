# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
    scale_transform,
    unscale_transform,
)

from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg


class HandoverEnv(DirectMARLEnv):
    cfg: HandoverEnvCfg

    def __init__(self, cfg: HandoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.right_hand.num_joints

        # buffers for position targets
        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.right_hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.right_hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.right_hand.data.joint_limits.torch.to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.0, -0.64, 0.54], device=self.device)
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # Sticky per-env flag: True once the object reached the goal within threshold.
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # unit tensors for sampling goal/object rotations about the x and y axes
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.right_hand = Articulation(self.cfg.right_robot_cfg)
        self.left_hand = Articulation(self.cfg.left_robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["right_robot"] = self.right_hand
        self.scene.articulations["left_robot"] = self.left_hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self._apply_hand_action(
            self.right_hand, "right_hand", self.right_hand_curr_targets, self.right_hand_prev_targets
        )
        self._apply_hand_action(self.left_hand, "left_hand", self.left_hand_curr_targets, self.left_hand_prev_targets)

    def _apply_hand_action(
        self,
        hand: Articulation,
        agent: str,
        curr_targets: torch.Tensor,
        prev_targets: torch.Tensor,
    ) -> None:
        """Map one agent's actions to joint position targets and write them to its hand.

        The raw ``[-1, 1]`` action is rescaled to the joint limits, blended with the previous
        target via the exponential moving average, clamped to the limits, and set on the hand.
        """
        idx = self.actuated_dof_indices
        lower = self.hand_dof_lower_limits[:, idx]
        upper = self.hand_dof_upper_limits[:, idx]

        targets = unscale_transform(self.actions[agent], lower, upper)
        targets = self.cfg.act_moving_average * targets + (1.0 - self.cfg.act_moving_average) * prev_targets[:, idx]
        targets = saturate(targets, lower, upper)

        curr_targets[:, idx] = targets
        prev_targets[:, idx] = targets
        hand.set_joint_position_target_index(target=targets, joint_ids=idx)

    def _hand_proprio_obs(self, agent: str) -> torch.Tensor:
        """Per-hand proprioceptive observation block for ``agent`` (133 dims).

        Layout: normalized DOF positions (24), scaled DOF velocities (24), fingertip positions
        (5*3), rotations (5*4), linear+angular velocities (5*6), and the applied actions (20).
        """
        side = agent.split("_")[0]  # "right" or "left"
        return torch.cat(
            (
                scale_transform(
                    getattr(self, f"{agent}_dof_pos"), self.hand_dof_lower_limits, self.hand_dof_upper_limits
                ),
                self.cfg.vel_obs_scale * getattr(self, f"{agent}_dof_vel"),
                getattr(self, f"{side}_fingertip_pos").view(self.num_envs, self.num_fingertips * 3),
                getattr(self, f"{side}_fingertip_rot").view(self.num_envs, self.num_fingertips * 4),
                getattr(self, f"{side}_fingertip_velocities").view(self.num_envs, self.num_fingertips * 6),
                self.actions[agent],
            ),
            dim=-1,
        )

    def _object_goal_obs(self) -> torch.Tensor:
        """Object and goal observation block shared by both agents and the critic state (24 dims).

        Layout: object position (3), rotation (4), linear velocity (3), scaled angular velocity (3),
        goal position (3), goal rotation (4), and the goal-to-object rotation difference (4).
        """
        return torch.cat(
            (
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.goal_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        object_goal = self._object_goal_obs()
        return {
            "right_hand": torch.cat((self._hand_proprio_obs("right_hand"), object_goal), dim=-1),
            "left_hand": torch.cat((self._hand_proprio_obs("left_hand"), object_goal), dim=-1),
        }

    def _get_states(self) -> torch.Tensor:
        return torch.cat(
            (self._hand_proprio_obs("right_hand"), self._hand_proprio_obs("left_hand"), self._object_goal_obs()),
            dim=-1,
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # compute reward
        goal_dist = torch.linalg.norm(self.object_pos - self.goal_pos, ord=2, dim=-1)
        rew_dist = 2 * torch.exp(-self.cfg.dist_reward_scale * goal_dist)

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist.mean()
        self.extras["log"]["Metrics/goal_distance"] = goal_dist.mean().item()
        # Sticky per-env success: True once the object reached the goal within threshold.
        self._episode_succeeded |= goal_dist < self.cfg.success_distance_threshold

        return {"right_hand": rew_dist, "left_hand": rew_dist}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()

        # reset when object has fallen
        out_of_reach = self.object_pos[:, 2] <= self.cfg.fall_dist
        # reset when episode ends
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_reach for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.right_hand._ALL_INDICES
        # Flush per-episode success (sticky binary: object ever reached the goal within threshold).
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._episode_succeeded[env_ids].float().mean().item()
        )
        self._episode_succeeded[env_ids] = False
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_pose = self.object.data.default_root_pose.torch.clone()[env_ids]
        object_default_vel = self.object.data.default_root_vel.torch.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        object_default_pose[:, 0:3] = (
            object_default_pose[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_pose[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_vel[:] = 0.0
        self.object.write_root_pose_to_sim_index(root_pose=object_default_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(root_velocity=object_default_vel, env_ids=env_ids)

        # reset right hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos.torch[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos.torch[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.right_hand.data.default_joint_pos.torch[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel.torch[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos

        self.right_hand.set_joint_position_target_index(target=dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_velocity_to_sim_index(velocity=dof_vel, env_ids=env_ids)

        # reset left hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos.torch[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos.torch[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.left_hand.data.default_joint_pos.torch[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel.torch[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos

        self.left_hand.set_joint_position_target_index(target=dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_velocity_to_sim_index(velocity=dof_vel, env_ids=env_ids)

        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):
        # data for right hand
        self.right_fingertip_pos = self.right_hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w.torch[:, self.finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos.torch
        self.right_hand_dof_vel = self.right_hand.data.joint_vel.torch

        # data for left hand
        self.left_fingertip_pos = self.left_hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w.torch[:, self.finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos.torch
        self.left_hand_dof_vel = self.left_hand.data.joint_vel.torch

        # data for object
        self.object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w.torch
        self.object_linvel = self.object.data.root_lin_vel_w.torch
        self.object_angvel = self.object.data.root_ang_vel_w.torch


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
