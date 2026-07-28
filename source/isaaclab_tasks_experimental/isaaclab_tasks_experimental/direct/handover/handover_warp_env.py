# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first Direct environment for the Shadow handover task.

Experimental counterpart of the torch-first mainline implementation in
:mod:`isaaclab_tasks.core.handover`; see the reorient variant for the conventions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg
from isaaclab_tasks.core.handover.handover_task_base import GOAL_POSITION_OFFSET
from isaaclab_tasks.core.utils import EpisodeErrorRecorder, randomize_rotation, sample_joint_positions_within_limits

from isaaclab_tasks_experimental.direct.handover.handover_kernels import (
    fall_kernel,
    hand_proprio_kernel,
    handover_reward_kernel,
    handover_success_kernel,
    object_goal_kernel,
)
from isaaclab_tasks_experimental.direct.reorient.reorient_kernels import ema_actuation_kernel


class HandoverWarpEnv(DirectMARLEnv):
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
        self.actuated_dof_indices, _ = self.right_hand.find_joints(cfg.actuated_joint_names)
        if len(self.actuated_dof_indices) != len(cfg.actuated_joint_names):
            raise ValueError(
                f"Expected {len(cfg.actuated_joint_names)} actuated joints, found {len(self.actuated_dof_indices)}."
            )

        # finger bodies
        self.finger_bodies, _ = self.right_hand.find_bodies(self.cfg.fingertip_body_names)
        if len(self.finger_bodies) != len(self.cfg.fingertip_body_names):
            raise ValueError(
                f"Expected {len(self.cfg.fingertip_body_names)} fingertip bodies, found {len(self.finger_bodies)}."
            )
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.right_hand.data.joint_limits.torch.to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # goal = object default position + shared offset (mirrors HandoverCommand.__init__)
        self.goal_pos[:, :] = self.object.data.default_root_pose.torch[:, :3].to(self.device) + torch.tensor(
            GOAL_POSITION_OFFSET, dtype=torch.float, device=self.device
        )
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # Sticky per-env flag: True once the object reached the goal within threshold.
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_distance = EpisodeErrorRecorder(self.num_envs, self.device)
        self._success_flags = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_distance_buf = torch.empty(self.num_envs, dtype=torch.float32, device=self.device)
        self._reward_buf = torch.empty(self.num_envs, dtype=torch.float32, device=self.device)
        # cached Warp views of the buffers above; the hot loop launches kernels without conversions
        self._success_flags_wp = wp.from_torch(self._success_flags)
        self._goal_distance_buf_wp = wp.from_torch(self._goal_distance_buf)
        self._reward_buf_wp = wp.from_torch(self._reward_buf)
        self._env_origins_wp = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)
        self._goal_pos_wp = wp.from_torch(self.goal_pos, dtype=wp.vec3f)
        self._goal_rot_wp = wp.from_torch(self.goal_rot, dtype=wp.quatf)
        self._object_goal_buf = torch.empty(self.num_envs, 24, dtype=torch.float32, device=self.device)
        self._object_goal_buf_wp = wp.from_torch(self._object_goal_buf)
        self._lower_limits_wp = wp.from_torch(self.hand_dof_lower_limits)
        self._upper_limits_wp = wp.from_torch(self.hand_dof_upper_limits)
        self._actuated_dof_ids_wp = wp.array(self.actuated_dof_indices, dtype=wp.int32, device=str(self.device))
        self._finger_ids_wp = wp.array(self.finger_bodies, dtype=wp.int32, device=str(self.device))
        self._hand_targets_wp = {
            "right_hand": (wp.from_torch(self.right_hand_prev_targets), wp.from_torch(self.right_hand_curr_targets)),
            "left_hand": (wp.from_torch(self.left_hand_prev_targets), wp.from_torch(self.left_hand_curr_targets)),
        }
        self._compact_targets = torch.zeros(
            (self.num_envs, len(self.actuated_dof_indices)), dtype=torch.float32, device=self.device
        )
        self._compact_targets_wp = wp.from_torch(self._compact_targets)
        self._fell_flags = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self._fell_flags_wp = wp.from_torch(self._fell_flags)
        num_joints, num_fingers = self.num_hand_dofs, self.num_fingertips
        proprio_dim = 2 * num_joints + 13 * num_fingers + self.cfg.action_spaces["right_hand"]
        self._proprio_bufs = {
            agent: torch.empty(self.num_envs, proprio_dim, device=self.device) for agent in ("right_hand", "left_hand")
        }
        self._proprio_bufs_wp = {agent: wp.from_torch(buf) for agent, buf in self._proprio_bufs.items()}

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
        self._apply_hand_action(self.right_hand, "right_hand")
        self._apply_hand_action(self.left_hand, "left_hand")

    def _apply_hand_action(self, hand: Articulation, agent: str) -> None:
        """Map one agent's actions to joint position targets and write them to its hand.

        The raw ``[-1, 1]`` action is rescaled to the joint limits, blended with the previous
        target via the exponential moving average, clamped to the limits, and set on the hand.
        """
        prev_wp, curr_wp = self._hand_targets_wp[agent]
        wp.launch(
            ema_actuation_kernel,
            dim=(self.num_envs, self._actuated_dof_ids_wp.shape[0]),
            inputs=[
                wp.from_torch(self.actions[agent]),
                self._lower_limits_wp,
                self._upper_limits_wp,
                self._actuated_dof_ids_wp,
                self.cfg.act_moving_average,
                prev_wp,
                curr_wp,
            ],
            outputs=[self._compact_targets_wp],
            device=self._compact_targets_wp.device,
        )
        hand.set_joint_position_target_index(target=self._compact_targets, joint_ids=self.actuated_dof_indices)

    def _hand_proprio_obs(self, agent: str) -> torch.Tensor:
        """Per-hand proprioceptive observation block for ``agent`` (133 dims).

        Layout: normalized DOF positions (24), scaled DOF velocities (24), fingertip positions
        (5*3), rotations (5*4), linear+angular velocities (5*6), and the applied actions (20).
        """
        hand = self.right_hand if agent == "right_hand" else self.left_hand
        wp.launch(
            hand_proprio_kernel,
            dim=(self.num_envs, self._proprio_bufs_wp[agent].shape[1]),
            inputs=[
                hand.data.joint_pos.warp,
                hand.data.joint_vel.warp,
                self._lower_limits_wp,
                self._upper_limits_wp,
                self.cfg.vel_obs_scale,
                hand.data.body_pos_w.warp,
                hand.data.body_quat_w.warp,
                hand.data.body_vel_w.warp,
                self._finger_ids_wp,
                self._env_origins_wp,
                wp.from_torch(self.actions[agent]),
            ],
            outputs=[self._proprio_bufs_wp[agent]],
            device=self._proprio_bufs_wp[agent].device,
        )
        return self._proprio_bufs[agent]

    def _object_goal_obs(self) -> torch.Tensor:
        """Object and goal observation block shared by both agents and the critic state (24 dims).

        Layout: object position (3), rotation (4), linear velocity (3), scaled angular velocity (3),
        goal position (3), goal rotation (4), and the goal-to-object rotation difference (4).
        """
        wp.launch(
            object_goal_kernel,
            dim=self.num_envs,
            inputs=[
                self.object.data.root_pos_w.warp,
                self._env_origins_wp,
                self.object.data.root_quat_w.warp,
                self.object.data.root_lin_vel_w.warp,
                self.object.data.root_ang_vel_w.warp,
                self._goal_pos_wp,
                self._goal_rot_wp,
                self.cfg.vel_obs_scale,
            ],
            outputs=[self._object_goal_buf_wp],
            device=self._object_goal_buf_wp.device,
        )
        return self._object_goal_buf

    def _get_observations(self) -> dict[str, torch.Tensor]:
        object_goal = self._object_goal_obs()
        return {
            "right_hand": torch.cat((self._hand_proprio_obs("right_hand"), object_goal), dim=-1),
            "left_hand": torch.cat((self._hand_proprio_obs("left_hand"), object_goal), dim=-1),
        }

    def _get_states(self) -> torch.Tensor:
        # DirectMARLEnv.step() and reset() always run _get_observations() immediately before
        # any state() call with no sim advance in between, so the proprio and object-goal
        # buffers already hold this step's values; reuse them instead of relaunching kernels
        return torch.cat(
            (self._proprio_bufs["right_hand"], self._proprio_bufs["left_hand"], self._object_goal_buf),
            dim=-1,
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # compute reward
        wp.launch(
            handover_success_kernel,
            dim=self.num_envs,
            inputs=[
                self.object.data.root_pos_w.warp,
                self._env_origins_wp,
                self._goal_pos_wp,
                self.cfg.success_distance_threshold,
            ],
            outputs=[self._success_flags_wp, self._goal_distance_buf_wp],
            device=self._success_flags_wp.device,
        )
        succeeded, goal_dist = self._success_flags, self._goal_distance_buf
        self._goal_distance.update(goal_dist)
        wp.launch(
            handover_reward_kernel,
            dim=self.num_envs,
            inputs=[self._goal_distance_buf_wp, self.cfg.dist_reward_scale],
            outputs=[self._reward_buf_wp],
            device=self._reward_buf_wp.device,
        )
        rew_dist = self._reward_buf

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        # tensors, not .item(): a host sync every step stalls the GPU at large env counts
        goal_dist_mean = goal_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist_mean
        self.extras["log"]["Metrics/goal_distance"] = goal_dist_mean
        # Sticky per-env success: True once the object reached the goal within threshold.
        self._episode_succeeded |= succeeded

        return {"right_hand": rew_dist, "left_hand": rew_dist}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # reset when object has fallen
        wp.launch(
            fall_kernel,
            dim=self.num_envs,
            inputs=[self.object.data.root_pos_w.warp, self._env_origins_wp, self.cfg.fall_dist],
            outputs=[self._fell_flags_wp],
            device=self._fell_flags_wp.device,
        )
        out_of_reach = self._fell_flags
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
        for statistic, value in self._goal_distance.reset(env_ids).items():
            self.extras["log"][f"Diagnostics/episode_min_goal_distance_{statistic}"] = value
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
        default_dof_pos = self.right_hand.data.default_joint_pos.torch[env_ids]
        dof_limits = self.right_hand.data.joint_limits.torch[env_ids]
        dof_pos = sample_joint_positions_within_limits(default_dof_pos, dof_limits, self.cfg.reset_dof_pos_noise)

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel.torch[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos

        self.right_hand.set_joint_position_target_index(target=dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_velocity_to_sim_index(velocity=dof_vel, env_ids=env_ids)

        # reset left hand
        default_dof_pos = self.left_hand.data.default_joint_pos.torch[env_ids]
        dof_limits = self.left_hand.data.joint_limits.torch[env_ids]
        dof_pos = sample_joint_positions_within_limits(default_dof_pos, dof_limits, self.cfg.reset_dof_pos_noise)

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel.torch[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos

        self.left_hand.set_joint_position_target_index(target=dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_position_to_sim_index(position=dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_velocity_to_sim_index(velocity=dof_vel, env_ids=env_ids)

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
