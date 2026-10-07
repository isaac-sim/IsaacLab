# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow two-hand handover environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_mul,
    sample_uniform,
    saturate,
    scale_transform,
    unscale_transform,
)

from isaaclab_tasks.core.reorient.utils import (
    EpisodeErrorRecorder,
    SuccessTracker,
    randomize_rotation,
    resolve_actuated_tendons,
    sample_joint_positions_within_limits,
)

from .mdp.rewards import evaluate_handover_success, handover_reward

if TYPE_CHECKING:
    from .handover_env_cfg import HandoverEnvCfg


class HandoverEnv(DirectMARLEnv):
    """Two Shadow Hands repeatedly hand a ball between alternating goal positions.

    Both agents observe their own hand plus the object and goal, and share one distance reward.
    """

    cfg: HandoverEnvCfg

    def __init__(self, cfg: HandoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.right_hand, self.left_hand, self.object, self.goal_markers = [
            self.scene[name] for name in ("right_robot", "left_robot", "object", "goal_object")
        ]

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

        # Motors that pull a tendon rather than drive a joint. Both hands are the same model, so
        # one index set serves both.
        self.actuated_tendon_indices = torch.empty(0, dtype=torch.long, device=self.device)
        if cfg.actuated_tendon_names:
            self.actuated_tendon_indices, self.tendon_lower_limits, self.tendon_upper_limits = resolve_actuated_tendons(
                self.right_hand,
                cfg.actuated_tendon_names,
                self.num_envs,
                self.device,
                cfg.actuated_tendon_position_limits,
            )

        # finger bodies
        self.finger_bodies, _ = self.right_hand.find_bodies(self.cfg.fingertip_body_names)
        if len(self.finger_bodies) != len(self.cfg.fingertip_body_names):
            raise ValueError(
                f"Expected {len(self.cfg.fingertip_body_names)} fingertip bodies, found {len(self.finger_bodies)}."
            )
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.right_hand.data.joint_limits.torch
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 3] = 1.0  # identity quaternion in (x, y, z, w) layout

        # One goal per hand, each the same offset in its OWN hand's frame, so the two are
        # mirror images. Resolved once here from the hands' default poses: the hands are
        # fixed bases, so re-deriving this per reset would recompute a constant.
        offset = torch.tensor(self.cfg.goal_position_offset, dtype=torch.float, device=self.device)
        offset = offset.unsqueeze(0).expand(self.num_envs, 3)
        goals = []
        for hand in (self.right_hand, self.left_hand):
            root_pose = hand.data.default_root_pose.torch
            goals.append(root_pose[:, :3] + quat_apply(root_pose[:, 3:7], offset))
        # [2, num_envs, 3] -- index 0 is the right hand's goal, 1 is the left hand's
        self._goal_pos_per_side = torch.stack(goals, dim=0)
        # Which hand the object is being delivered TO. The object spawns at the right
        # hand, so the first goal is the left hand's.
        self._goal_side = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._env_arange = torch.arange(self.num_envs, device=self.device)
        self.goal_pos = self._goal_pos_per_side[self._goal_side, self._env_arange]
        # Cumulative steps inside the threshold for the current goal attempt.
        self._dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Counts banked goals per episode and turns them into goals/(goals+1).
        self._success = SuccessTracker(self.num_envs, self.device)

        self._goal_distance = EpisodeErrorRecorder(self.num_envs, self.device)

        # unit tensors for sampling goal/object rotations about the x and y axes
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

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
        """Map one agent's actions to position targets and write them to its hand.

        Actions are ordered joints first, then tendons, matching the manager task's action term.
        Each raw ``[-1, 1]`` joint action is rescaled to the joint limits, blended with the
        previous target via the exponential moving average, clamped to the limits, and set on the
        hand. Tendon actions are rescaled to the tendon's commandable range and written directly.
        """
        idx = self.actuated_dof_indices
        lower = self.hand_dof_lower_limits[:, idx]
        upper = self.hand_dof_upper_limits[:, idx]

        targets = unscale_transform(self.actions[agent][:, : len(idx)], lower, upper)
        targets = self.cfg.act_moving_average * targets + (1.0 - self.cfg.act_moving_average) * prev_targets[:, idx]
        targets = saturate(targets, lower, upper)

        curr_targets[:, idx] = targets
        prev_targets[:, idx] = targets
        hand.set_joint_position_target_index(target=targets, joint_ids=idx)

        if len(self.actuated_tendon_indices) > 0:
            # No moving average on the tendon target: the manager task's action term applies none,
            # and the two task variants have to stay comparable.
            # saturate like the joint target above: a Gaussian policy samples past the action range,
            # and the manager term bounds its own output, so both variants must clamp to the limits
            tendon_targets = unscale_transform(
                self.actions[agent][:, len(idx) :], self.tendon_lower_limits, self.tendon_upper_limits
            )
            hand.set_fixed_tendon_position_target_index(
                target=saturate(tendon_targets, self.tendon_lower_limits, self.tendon_upper_limits),
                fixed_tendon_ids=self.actuated_tendon_indices,
            )

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
        succeeded, goal_dist = evaluate_handover_success(
            self.object_pos, self.goal_pos, self.cfg.success_distance_threshold
        )
        self._goal_distance.update(goal_dist)
        rew_dist = handover_reward(goal_dist, self.cfg.dist_reward_scale)

        # log as tensors, not .item(): a per-step host sync stalls the GPU
        if "log" not in self.extras:
            self.extras["log"] = dict()
        goal_dist_mean = goal_dist.mean()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist_mean
        self.extras["log"]["Metrics/goal_distance"] = goal_dist_mean

        # A goal is banked only after the object has spent ``success_dwell_steps`` inside the
        # threshold. The tally is cumulative over the attempt, so a momentary exit does not
        # erase the progress, but a single touch in passing is not enough either.
        self._dwell += succeeded.long()
        # ``earned`` drops the goal a reset handed out and releases its own guard, so it
        # has to run every step, not only when something banked.
        banked = self._success.earned(self._dwell >= self.cfg.success_dwell_steps)
        banked_ids = banked.nonzero(as_tuple=False).squeeze(-1)
        if banked_ids.numel() > 0:
            # Hand the object back: the goal moves to the other hand, so the receiver of this
            # goal becomes the thrower for the next one.
            self._success.record_goal_reached(banked_ids)
            self._goal_side[banked_ids] = 1 - self._goal_side[banked_ids]
            self._dwell[banked_ids] = 0
            self._refresh_goal_pos()

        self.extras["log"]["Diagnostics/dwell_steps"] = self._dwell.float().mean()

        return {"right_hand": rew_dist, "left_hand": rew_dist}

    def _refresh_goal_pos(self) -> None:
        """Point ``goal_pos`` at the currently targeted hand and redraw the marker."""
        self.goal_pos = self._goal_pos_per_side[self._goal_side, self._env_arange]
        self.goal_markers.visualize(
            self.goal_pos + self.scene.env_origins,
            self.goal_rot,
            environment_ids=self.scene._ALL_INDICES,
        )

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
        # Goals are banked during the episode, so success is the count the episode reached
        # rather than where the object sits at the final step. Snapshot BEFORE the reset
        # below re-points the goal, or the new episode's goal is credited to the old one.
        goals = self._success.snapshot(env_ids)
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (goals / (goals + 1.0)).mean()
        self.extras["log"]["Metrics/consecutive_success"] = goals.mean()
        for statistic, value in self._goal_distance.reset(env_ids).items():
            self.extras["log"][f"Diagnostics/episode_min_goal_distance_{statistic}"] = value
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

        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # A new episode starts with the object at the right hand, so it is delivered to the
        # left first; the alternation then runs from there. Dwell restarts with it.
        self._goal_side[env_ids] = 1
        self._dwell[env_ids] = 0
        # Rewards are evaluated before autoreset, so the next physics step can earn a goal.
        self._success.clear(env_ids, skip_next_update=torch.zeros(len(env_ids), dtype=torch.bool, device=self.device))

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        self._refresh_goal_pos()

    def _compute_intermediate_values(self) -> None:
        env_origins = self.scene.env_origins.unsqueeze(1)
        # data for right hand
        self.right_fingertip_pos = self.right_hand.data.body_pos_w.torch[:, self.finger_bodies] - env_origins
        self.right_fingertip_rot = self.right_hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w.torch[:, self.finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos.torch
        self.right_hand_dof_vel = self.right_hand.data.joint_vel.torch

        # data for left hand
        self.left_fingertip_pos = self.left_hand.data.body_pos_w.torch[:, self.finger_bodies] - env_origins
        self.left_fingertip_rot = self.left_hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w.torch[:, self.finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos.torch
        self.left_hand_dof_vel = self.left_hand.data.joint_vel.torch

        # data for object
        self.object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w.torch
        self.object_linvel = self.object.data.root_lin_vel_w.torch
        self.object_angvel = self.object.data.root_ang_vel_w.torch
