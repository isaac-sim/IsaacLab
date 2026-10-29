# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical deliveries and success-guided reset states for suction parcel sorting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, EventTermCfg, ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from isaaclab_tasks.utils.reset_states import ResetStateSampler, ResetStateSamplerCfg

from ..conveyor_franka.mdp.kinematics import end_effector_pose, tool_velocity
from ..conveyor_franka.mdp.terminations import invalid_action as invalid_policy_action
from .scene import (
    BELT_TOP_Z,
    BIN_INTERIOR_SIZE,
    BIN_POSITIONS,
    GOAL_ANGLES,
    GOAL_DIRECTIONS,
    GOAL_POSITIONS,
    PARCEL_CLASS_IDS,
    PARCEL_SIZES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def parcel_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observe the active package in the tool frame, with dimensions and relative velocities."""
    state = _active_state(env)
    position, rotation = end_effector_pose(env)
    inverse = math_utils.quat_conjugate(rotation)
    velocity = state[:, 7:13] - tool_velocity(env)
    return torch.cat(
        (
            math_utils.quat_apply(inverse, state[:, :3] - position),
            math_utils.quat_mul(inverse, state[:, 3:7]),
            math_utils.quat_apply(inverse, velocity[:, :3]),
            math_utils.quat_apply(inverse, velocity[:, 3:]),
            env.parcel_sizes[env.active_ids],
        ),
        dim=-1,
    )


def tool_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observe tool pose, velocity, and the controller's remaining Cartesian target error."""
    position, rotation = end_effector_pose(env)
    arm = env.action_manager.get_term("arm_action")
    target_error = arm._ik_controller.ee_pos_des - arm._compute_frame_pose()[0]
    target_error = torch.where(arm._needs_reference[:, None], 0.0, target_error)
    return torch.cat((position - env.scene.env_origins, rotation, tool_velocity(env), target_error), dim=-1)


def suction_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observe physical attachment and the active parcel's pickup qualification."""
    command = env.command_manager.get_term("sort")
    return torch.stack((env.gripper.engaged, command.lifted), dim=-1).float()


def successful_delivery(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End an episode when its configured sorted-inventory goal is met."""
    command = env.command_manager.get_term("sort")
    command.evaluate()
    return env.episode_success


def failed_delivery(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Detect wrong bins, dropped active packages, and nonfinite physical state."""
    command = env.command_manager.get_term("sort")
    command.evaluate()
    return command.failed


@configclass
class SizeSortCommandCfg(CommandTermCfg):
    """Select active packages and verify their size-specific physical destinations."""

    class_type: type | str = "{DIR}.mdp:SizeSortCommand"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    reset_event_name: str = "reset_parcels"
    deliveries_per_episode: int = 12
    """Sorted-parcel goal; full-batch reset recipes can include pre-sorted inventory."""


class SizeSortCommand(CommandTerm):
    """Verify released packages inside the correct bins after an actual pickup."""

    cfg: SizeSortCommandCfg

    def __init__(self, cfg: SizeSortCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if not 1 <= cfg.deliveries_per_episode <= len(env.parcels):
            raise ValueError("deliveries_per_episode must be between one and the number of parcels.")
        self._reset_term = env.event_manager.get_term_cfg(cfg.reset_event_name).func
        self._ids = torch.arange(self.num_envs, device=self.device)
        self.delivered = torch.zeros((self.num_envs, len(env.parcels)), dtype=torch.bool, device=self.device)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.full_start = torch.zeros_like(self.lifted)
        self.failed = torch.zeros_like(self.lifted)
        self.new_success = torch.zeros_like(self.lifted)
        self._waiting = torch.zeros_like(self.lifted)
        self._stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_step = torch.full_like(self._stable_steps, -1)
        self._evaluation_step = -1
        self._bin_frames = torch.zeros((5, 3, 3), device=self.device)
        self._bin_frames[:, 0] = env.bin_directions
        self._bin_frames[:, 1, 0] = -env.bin_directions[:, 1]
        self._bin_frames[:, 1, 1] = env.bin_directions[:, 0]
        self._bin_frames[:, 2, 2] = 1.0
        self._bin_half_size = torch.tensor(BIN_INTERIOR_SIZE, device=self.device) / 2.0
        for name in ("success_rate", "correct_deliveries", "wrong_deliveries", "full_batch_success_rate"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Dimensions of the selected parcel [m], shape [num_envs, 3]."""
        return self._env.parcel_sizes[self._env.active_ids]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, torch.Tensor]:
        """Report only the resetting episodes; preserve all other environments' counters."""
        ids = self._ids[slice(None) if env_ids is None else env_ids]
        full = self.full_start[ids] & (self._env.episode_length_buf[ids] > 0)
        full_ids = ids[full]
        full_success = self.metrics["success_rate"][full_ids]
        full_batch = self.metrics["full_batch_success_rate"][full_ids]
        extras = super().reset(ids)
        extras.pop("full_batch_success_rate")
        if full_ids.numel():
            # RSL concatenates episode values; scalar reset-batch means misweight this conditional rate.
            extras["full_start_success_rate"] = full_success
            extras["full_batch_success_rate"] = full_batch
        extras["full_start_episodes"] = full.sum()
        extras["reset_row_coverage"] = (self._reset_term.reset_sampler.success_monitor.success_size > 0).float().mean()
        extras["full_start_sampling_probability"] = self._reset_term.reset_sampler.probabilities[
            self._reset_term.phase_ids == 3
        ].sum()
        return extras

    def evaluate(self) -> None:
        """Evaluate physical receipts once per environment policy step."""
        env = self._env
        if self._evaluation_step == env.common_step_counter:
            return
        changed = self._last_step != env.episode_length_buf
        states = _parcel_states(env)
        state = states[self._ids, env.active_ids]
        local = state[:, :3] - env.scene.env_origins
        self.lifted |= changed & ~self._waiting & env.gripper.engaged & (local[:, 2] > BELT_TOP_Z + 0.10)
        offsets = local[:, None] - env.bin_positions
        in_bin_frame = torch.einsum("kij,nkj->nki", self._bin_frames, offsets)
        rotation = math_utils.matrix_from_quat(state[:, 3:7])
        half_extents = (
            torch.einsum("kij,njl->nkil", self._bin_frames, rotation).abs()
            * (env.parcel_sizes[env.active_ids, None, None] / 2.0)
        ).sum(dim=-1)
        inside = (in_bin_frame.abs() + half_extents < self._bin_half_size + 0.005).all(dim=-1)
        over_bin = (in_bin_frame.abs() < self._bin_half_size + 0.03).all(dim=-1)
        released = ~env.gripper.engaged
        settled = torch.linalg.vector_norm(state[:, 7:10], dim=-1) < 0.35
        correct = inside[self._ids, env.parcel_classes[env.active_ids]]
        eligible = self.lifted & released & settled & ~self._waiting
        stable = eligible & correct
        self._stable_steps = torch.where(changed, torch.where(stable, self._stable_steps + 1, 0), self._stable_steps)
        completed = (self._stable_steps >= 3) & ~self.delivered[self._ids, env.active_ids]
        self.new_success = torch.where(changed, completed, self.new_success)
        self.delivered[self._ids, env.active_ids] |= self.new_success
        wrong = eligible & inside.any(dim=-1) & ~correct
        fallen = (states[:, :, 2] - env.scene.env_origins[:, None, 2] < -0.65) & ~self.delivered
        correct_bin = over_bin[self._ids, env.parcel_classes[env.active_ids]]
        fallen[self._ids, env.active_ids] &= ~(self.lifted & released & correct_bin)
        finite = torch.isfinite(states).all(dim=(1, 2)) & torch.isfinite(env.scene["robot"].data.joint_pos.torch).all(
            dim=-1
        )
        self.failed |= changed & (wrong | fallen.any(dim=-1) | ~finite | invalid_policy_action(env))
        count = self.delivered.sum(dim=-1)
        env.episode_success[:] = (count >= self.cfg.deliveries_per_episode) & ~self.failed
        self.metrics["success_rate"][:] = env.episode_success.float()
        initial_count = (
            self._reset_term.completed_counts[self._reset_term.row_ids]
            if self.cfg.deliveries_per_episode == len(env.parcels)
            else 0
        )
        self.metrics["correct_deliveries"][:] = (count - initial_count).float()
        self.metrics["wrong_deliveries"][changed & wrong] = 1.0
        self.metrics["full_batch_success_rate"][:] = (
            self.full_start & self.delivered.all(dim=-1) & ~self.failed
        ).float()
        self._last_step.copy_(env.episode_length_buf)
        self._evaluation_step = env.common_step_counter
        env.extras["successes"] = env.episode_success.clone()

    def _update_metrics(self) -> None:
        self.evaluate()

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        ids = self._ids[env_ids]
        phase = self._reset_term.phase_ids[self._reset_term.row_ids[ids]]
        self.delivered[ids] = (
            self._reset_term.completed[self._reset_term.row_ids[ids]]
            if self.cfg.deliveries_per_episode == len(self._env.parcels)
            else False
        )
        self.lifted[ids] = phase < 2
        self.full_start[ids] = phase == 3
        self.failed[ids] = self.new_success[ids] = self._waiting[ids] = False
        self._stable_steps[ids] = 0
        self._last_step[ids] = -1
        self._evaluation_step = -1

    def _update_command(self) -> None:
        if self.cfg.deliveries_per_episode == 1:
            return
        self._waiting |= self.new_success
        ids = self._waiting.nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        positions = torch.stack([parcel.data.root_pos_w.torch[ids] for parcel in self._env.parcels], dim=1)
        local = positions - self._env.scene.env_origins[ids, None]
        reachable = (
            (local[:, :, 0] > 0.38) & (local[:, :, 0] < 0.85) & (local[:, :, 1] > -0.31) & (local[:, :, 1] < -0.07)
        )
        eligible = reachable & ~self.delivered[ids]
        tool, _ = end_effector_pose(self._env)
        distances = torch.linalg.vector_norm(positions - tool[ids, None], dim=-1)
        choice = distances.masked_fill(~eligible, torch.inf).argmin(dim=-1)
        available = eligible.any(dim=-1)
        selected = ids[available]
        self._env.active_ids[selected] = choice[available]
        self.lifted[selected] = self._waiting[selected] = False
        self.new_success[selected] = False
        self._stable_steps[selected] = 0
        self._env.gripper.reset(selected)


class ResetParcels(ManagerTermBase):
    """Restore a fixed Cartesian reset catalog and sample it with shared outcome history."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        from .actions import solve_tool_ik

        super().__init__(cfg, env)
        rows = torch.arange(len(PARCEL_SIZES) * 8, device=self.device)
        self.row_count = rows.numel()
        self.parcel_ids = rows // 8
        self.phase_ids = (rows // 2) % 4
        variants = rows % 2
        self.reset_sampler = ResetStateSampler(cfg.params["sampling"], self.num_envs, self.row_count, self.device)
        self.row_ids = self.reset_sampler.row_ids
        self.held_rows = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._initial_poses = torch.stack(
            [env.scene[f"parcel_{index}"].data.default_root_pose.torch[0] for index in range(len(PARCEL_SIZES))]
        )
        self.completed_counts = torch.tensor((11, 8, 4, 0), device=self.device)[self.phase_ids]
        rank = (torch.arange(len(PARCEL_SIZES), device=self.device) - self.parcel_ids[:, None] - 1) % len(PARCEL_SIZES)
        self.completed = rank < self.completed_counts[:, None]
        self._bin_poses = self._initial_poses.clone()
        directions = torch.tensor(GOAL_DIRECTIONS, device=self.device)
        bins = torch.tensor(BIN_POSITIONS, device=self.device)
        for index, class_id in enumerate(PARCEL_CLASS_IDS):
            rank_in_bin = PARCEL_CLASS_IDS[:index].count(class_id)
            radial = -0.12 if rank_in_bin % 2 == 0 else 0.12
            lateral = -0.10 if rank_in_bin < 2 else 0.10
            direction = directions[class_id]
            cross = torch.stack((-direction[1], direction[0], direction[2]))
            self._bin_poses[index, :3] = bins[class_id] + radial * direction + lateral * cross
            self._bin_poses[index, 2] += -BIN_INTERIOR_SIZE[2] / 2 + PARCEL_SIZES[index][2] / 2 + 0.006
            angle = torch.tensor(GOAL_ANGLES[class_id], device=self.device)
            self._bin_poses[index, 5:7] = torch.stack((torch.sin(angle / 2), torch.cos(angle / 2)))
        self._arm_ids = env.scene["robot"].find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        sizes = torch.tensor(PARCEL_SIZES, device=self.device)[self.parcel_ids]
        pick = torch.zeros((self.row_count, 3), device=self.device)
        pick[:, 0], pick[:, 1], pick[:, 2] = 0.54 + 0.10 * variants, -0.19, BELT_TOP_Z + sizes[:, 2] + 0.004
        classes = torch.tensor(PARCEL_CLASS_IDS, device=self.device)[self.parcel_ids]
        goal = torch.tensor(GOAL_POSITIONS, device=self.device)[classes]
        goal[:, 2] += sizes[:, 2] + 0.025
        carry = torch.lerp(pick, goal, (0.35 + 0.35 * variants)[:, None])
        carry[:, 2] = 0.30 + 0.05 * variants
        self._tool_positions = torch.where(
            (self.phase_ids == 0)[:, None], goal, torch.where((self.phase_ids == 1)[:, None], carry, pick)
        )
        pregrasp = (self.phase_ids == 2) & (variants == 1)
        self._tool_positions[pregrasp, 0] -= 0.02
        self._tool_positions[pregrasp, 2] += 0.07
        self._arm_positions = torch.tensor((0.0, -0.35, 0.0, -2.35, 0.0, 2.0, 0.0), device=self.device).repeat(
            self.row_count, 1
        )
        partial = self.phase_ids < 3
        self._arm_positions[partial] = solve_tool_ik(self._tool_positions[partial])
        self._pick_positions = pick.clone()
        self._pick_positions[:, 2] -= 0.5 * sizes[:, 2] + 0.002

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        sampling: ResetStateSamplerCfg,
        full_starts: bool = False,
    ) -> None:
        """Record completed outcomes before restoring only the selected environments."""
        del sampling
        ids = env.scene._ALL_INDICES[env_ids] if isinstance(env_ids, slice) else env_ids
        rows = self.reset_sampler.reset(
            ids,
            env.episode_success[ids],
            valid=env.episode_length_buf[ids] > 0,
            row_mask=self.phase_ids == 3 if full_starts else None,
        )
        active = self.parcel_ids[rows]
        env.active_ids[ids] = active
        env.episode_success[ids] = False
        env.gripper.reset(ids)
        phase = self.phase_ids[rows]
        self.held_rows[ids] = phase < 2
        batch = torch.arange(len(ids), device=self.device)
        slots = torch.rand((len(ids), len(env.parcels)), device=self.device).argsort(dim=1)
        slot_zero = slots.argmin(dim=1)
        slots[batch, slot_zero] = slots[batch, active]
        slots[batch, active] = 0
        positions = self._initial_poses[slots, :3]
        positions[:, :, 2] = BELT_TOP_Z + 0.5 * env.parcel_sizes[:, 2] + 0.002
        positions[:, :, 2] = torch.where(slots == len(env.parcels) - 1, self._initial_poses[-1, 2], positions[:, :, 2])
        active_position = self._pick_positions[rows].clone()
        active_position[:, 0] += (torch.rand(len(ids), device=self.device) - 0.5) * 0.06 * (phase == 3)
        held_position = self._tool_positions[rows].clone()
        held_position[:, 2] -= 0.5 * env.parcel_sizes[active, 2] + 0.002
        active_position = torch.where(self.held_rows[ids, None], held_position, active_position)
        positions[batch, active] = active_position
        yaw = (torch.rand((len(ids), len(env.parcels)), device=self.device) - 0.5) * 0.2
        velocity = torch.zeros((len(ids), 6), device=self.device)
        completed = self.completed[rows] if env.cfg.commands.sort.deliveries_per_episode == len(env.parcels) else None
        for index, parcel in enumerate(env.parcels):
            pose = self._initial_poses[index].expand(len(ids), -1).clone()
            pose[:, :3] = positions[:, index]
            pose[:, 5], pose[:, 6] = torch.sin(0.5 * yaw[:, index]), torch.cos(0.5 * yaw[:, index])
            if completed is not None:
                pose = torch.where(completed[:, index, None], self._bin_poses[index], pose)
            pose[:, :3] += env.scene.env_origins[ids]
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
        robot = env.scene["robot"]
        root = robot.data.default_root_pose.torch[ids].clone()
        root[:, :3] += env.scene.env_origins[ids]
        robot.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
        joints = robot.data.default_joint_pos.torch[ids].clone()
        joints[:, self._arm_ids] = self._arm_positions[rows]
        robot.write_joint_position_to_sim_index(position=joints, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), env_ids=ids)
        robot.actuators.target_command.set_position_index(value=joints, env_ids=ids)


def _active_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gather the selected physical parcel without changing its catalog identity."""
    states = _parcel_states(env)
    return states[env.scene._ALL_INDICES, env.active_ids]


def _parcel_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read link poses and center-of-mass velocities through the current asset API."""
    poses = torch.stack([parcel.data.root_link_pose_w.torch for parcel in env.parcels], dim=1)
    velocities = torch.stack([parcel.data.root_com_vel_w.torch for parcel in env.parcels], dim=1)
    return torch.cat((poses, velocities), dim=-1)
