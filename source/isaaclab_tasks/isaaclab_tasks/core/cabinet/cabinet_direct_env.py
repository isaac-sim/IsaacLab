# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, sample_uniform

if TYPE_CHECKING:
    from isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg import CabinetDirectEnvCfg


def _env_local_pose(prim_path: str, env_pos: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Read a prim's rest pose from the stage, expressed relative to its environment origin.

    Returns:
        The pose as ``[pos(3), quat_xyzw(4)]``.
    """
    xformable = UsdGeom.Xformable(get_current_stage().GetPrimAtPath(prim_path))
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    position = world_transform.ExtractTranslation()
    quat = world_transform.ExtractRotationQuat()
    return torch.tensor(
        [
            position[0] - env_pos[0],
            position[1] - env_pos[1],
            position[2] - env_pos[2],
            *quat.imaginary,
            quat.real,
        ],
        device=device,
    )


class CabinetDirectEnv(DirectRLEnv):
    """Direct-workflow environment for opening a cabinet drawer with a parallel-jaw gripper."""

    cfg: CabinetDirectEnvCfg

    def __init__(self, cfg: CabinetDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint limits and the per-joint fraction of the action scale applied each step
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits.torch[0, :, 0]
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits.torch[0, :, 1]
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        finger_dof_idx, _ = self._robot.find_joints(self.cfg.finger_joint_names)
        self.robot_dof_speed_scales[finger_dof_idx] = self.cfg.finger_speed_scale
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # The grasp frame sits midway between the fingertips, expressed in the hand frame. It is read
        # off the stage once at startup, since it is a fixed property of the gripper geometry.
        env_origin = self.scene.env_origins[0]
        robot_prim = "/World/envs/env_0/Robot/{}"
        hand_pose = _env_local_pose(robot_prim.format(self.cfg.hand_body_name), env_origin, self.device)
        lfinger_pose = _env_local_pose(robot_prim.format(self.cfg.left_finger_body_name), env_origin, self.device)
        rfinger_pose = _env_local_pose(robot_prim.format(self.cfg.right_finger_body_name), env_origin, self.device)

        finger_pos = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        hand_pose_inv_rot = quat_conjugate(hand_pose[3:7])
        hand_pose_inv_pos = -quat_apply(hand_pose_inv_rot, hand_pose[0:3])
        grasp_pos, grasp_rot = combine_frame_transforms(
            hand_pose_inv_pos, hand_pose_inv_rot, finger_pos, lfinger_pose[3:7]
        )
        grasp_pos = grasp_pos + torch.tensor(self.cfg.grasp_pos_offset, device=self.device)
        self.robot_local_grasp_pos = grasp_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = grasp_rot.repeat((self.num_envs, 1))

        drawer_grasp_pose = torch.tensor(self.cfg.drawer_local_grasp_pose, device=self.device)
        self.drawer_local_grasp_pos = drawer_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_grasp_pose[3:7].repeat((self.num_envs, 1))

        # axes used to score how well the gripper is oriented towards the drawer
        def _axis(vec: tuple[float, float, float]) -> torch.Tensor:
            return torch.tensor(vec, device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        self.gripper_forward_axis = _axis((0.0, 0.0, 1.0))
        self.gripper_up_axis = _axis((0.0, 1.0, 0.0))
        self.drawer_inward_axis = _axis((-1.0, 0.0, 0.0))
        self.drawer_up_axis = _axis((0.0, 0.0, 1.0))

        self.hand_link_idx = self._robot.find_bodies(self.cfg.hand_body_name)[0][0]
        self.left_finger_link_idx = self._robot.find_bodies(self.cfg.left_finger_body_name)[0][0]
        self.right_finger_link_idx = self._robot.find_bodies(self.cfg.right_finger_body_name)[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies(self.cfg.drawer_body_name)[0][0]
        self.drawer_joint_idx = self._cabinet.find_joints(self.cfg.drawer_joint_name)[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # sticky per-env flag: True once the drawer was opened past the success threshold
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)

        # PhysX replication requires explicit collision filtering between environments.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = (
            self.robot_dof_targets + self.robot_dof_speed_scales * self.step_dt * self.actions * self.cfg.action_scale
        )
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        opened = self._cabinet.data.joint_pos.torch[:, self.drawer_joint_idx] > self.cfg.termination_drawer_pos
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return opened, time_out

    def _get_rewards(self) -> torch.Tensor:
        # refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        drawer_pos = self._cabinet.data.joint_pos.torch[:, self.drawer_joint_idx]
        self._episode_succeeded |= drawer_pos > self.cfg.success_drawer_pos_threshold
        return self._compute_rewards(drawer_pos)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # flush per-episode success (sticky binary: drawer ever opened past the cfg threshold)
        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
        log["Metrics/drawer_pos"] = self._cabinet.data.joint_pos.torch[env_ids, self.drawer_joint_idx].mean().item()
        self._episode_succeeded[env_ids] = False

        super()._reset_idx(env_ids)

        # robot state, randomized around the default pose and clamped back into the joint limits
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids] + sample_uniform(
            *self.cfg.initial_joint_pos_range,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # cabinet state, fully closed
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_position_to_sim_index(position=zeros, env_ids=env_ids)
        self._cabinet.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=env_ids)

        # refresh the intermediate values so that _get_observations() sees the reset state
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos.torch - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel.torch * self.cfg.dof_velocity_scale,
                self.drawer_grasp_pos - self.robot_grasp_pos,
                self._cabinet.data.joint_pos.torch[:, self.drawer_joint_idx].unsqueeze(-1),
                self._cabinet.data.joint_vel.torch[:, self.drawer_joint_idx].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        hand_pos = self._robot.data.body_pos_w.torch[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w.torch[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w.torch[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w.torch[env_ids, self.drawer_link_idx]

        self.robot_grasp_pos[env_ids], self.robot_grasp_rot[env_ids] = combine_frame_transforms(
            hand_pos, hand_rot, self.robot_local_grasp_pos[env_ids], self.robot_local_grasp_rot[env_ids]
        )
        self.drawer_grasp_pos[env_ids], self.drawer_grasp_rot[env_ids] = combine_frame_transforms(
            drawer_pos, drawer_rot, self.drawer_local_grasp_pos[env_ids], self.drawer_local_grasp_rot[env_ids]
        )

    def _compute_rewards(self, drawer_pos: torch.Tensor) -> torch.Tensor:
        # distance from the grasp frame to the drawer handle
        distance = torch.linalg.norm(self.robot_grasp_pos - self.drawer_grasp_pos, ord=2, dim=-1)
        dist_reward = (1.0 / (1.0 + distance**2)) ** 2
        dist_reward = torch.where(distance <= 0.02, dist_reward * 2.0, dist_reward)

        # alignment of the gripper's forward and up axes with the drawer's
        forward_alignment = torch.sum(
            quat_apply(self.robot_grasp_rot, self.gripper_forward_axis)
            * quat_apply(self.drawer_grasp_rot, self.drawer_inward_axis),
            dim=-1,
        )
        up_alignment = torch.sum(
            quat_apply(self.robot_grasp_rot, self.gripper_up_axis)
            * quat_apply(self.drawer_grasp_rot, self.drawer_up_axis),
            dim=-1,
        )
        rot_reward = 0.5 * (
            torch.sign(forward_alignment) * forward_alignment**2 + torch.sign(up_alignment) * up_alignment**2
        )

        # penalty for a finger ending up on the wrong side of the handle
        lfinger_pos = self._robot.data.body_pos_w.torch[:, self.left_finger_link_idx]
        rfinger_pos = self._robot.data.body_pos_w.torch[:, self.right_finger_link_idx]
        lfinger_dist = lfinger_pos[:, 2] - self.drawer_grasp_pos[:, 2]
        rfinger_dist = self.drawer_grasp_pos[:, 2] - rfinger_pos[:, 2]
        finger_dist_penalty = lfinger_dist.clamp(max=0.0) + rfinger_dist.clamp(max=0.0)

        action_penalty = torch.sum(self.actions**2, dim=-1)

        rewards = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.rot_reward_scale * rot_reward
            + self.cfg.open_reward_scale * drawer_pos
            + self.cfg.finger_reward_scale * finger_dist_penalty
            - self.cfg.action_penalty_scale * action_penalty
        )
        # staged bonus for opening the drawer further
        for threshold in self.cfg.open_bonus_thresholds:
            rewards = torch.where(drawer_pos > threshold, rewards + self.cfg.open_bonus, rewards)

        self.extras.setdefault("log", {}).update(
            {
                "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
                "rot_reward": (self.cfg.rot_reward_scale * rot_reward).mean(),
                "open_reward": (self.cfg.open_reward_scale * drawer_pos).mean(),
                "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
                "finger_dist_penalty": (self.cfg.finger_reward_scale * finger_dist_penalty).mean(),
            }
        )
        return rewards
