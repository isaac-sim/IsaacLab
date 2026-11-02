# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical deliveries and success-guided reset states for suction parcel sorting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as common_mdp
from isaaclab.managers import CommandTerm, CommandTermCfg, EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from isaaclab_tasks.utils.reset_states import ResetStateSampler, ResetStateSamplerCfg

from ..conveyor_franka.mdp.kinematics import end_effector_pose, tool_velocity
from ..conveyor_franka.mdp.reset_events import franka_tool_position
from ..conveyor_franka.mdp.terminations import invalid_action as invalid_policy_action
from .actions_cfg import LINK7_TO_TOOL_Z
from .scene import (
    BELT_TOP_Z,
    BIN_INTERIOR_SIZE,
    GOAL_DIRECTIONS,
    GOAL_POSITIONS,
    INTAKE_CENTER,
    INTAKE_HALF_STRAIGHT,
    INTAKE_RADIUS,
    PARCEL_CLASS_IDS,
    PARCEL_SIZES,
    RECEIVING_PAD_HALF_WIDTH,
    RECEIVING_PAD_RADIAL_BOUNDS,
    SUCTION_PAD_RADIUS,
    SUCTION_PAD_SPACING,
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
    """Observe attachment, pickup history, and controlled-placement qualification."""
    command = env.command_manager.get_term("sort")
    return torch.stack((env.gripper.engaged, command.lifted, command.placed), dim=-1).float()


def batch_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Replenish the scene after every parcel has reached its bin."""
    command = env.command_manager.get_term("sort")
    command.evaluate()
    return command.delivered.all(dim=-1) & ~command.failed


def failed_delivery(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Detect wrong bins, dropped active packages, and nonfinite physical state."""
    command = env.command_manager.get_term("sort")
    command.evaluate()
    return command.failed


@configclass
class SizeSortCommandCfg(CommandTermCfg):
    """Command a controlled placement before a physical receipt at the selected outlet."""

    class_type: type | str = "{DIR}.mdp:SizeSortCommand"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    reset_event_name: str = "reset_parcels"
    randomize_destination: bool = True
    goal_radial_range: tuple[float, float] = (-0.02, 0.02)
    goal_lateral_range: tuple[float, float] = (-0.015, 0.015)
    placement_xy_tolerance: float = 0.035
    placement_max_height: float = 0.02
    placement_max_horizontal_speed: float = 0.15
    placement_max_vertical_speed: float = 0.10
    placement_max_angular_speed: float = 1.0
    placement_upright_tolerance: float = 0.15
    arm_contact_force_threshold: float = 5.0


class SizeSortCommand(CommandTerm):
    """Verify released packages inside the correct bins after an actual pickup."""

    cfg: SizeSortCommandCfg

    def __init__(self, cfg: SizeSortCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._reset_term = env.event_manager.get_term_cfg(cfg.reset_event_name).func
        self._ids = torch.arange(self.num_envs, device=self.device)
        self.delivered = torch.zeros((self.num_envs, len(env.parcels)), dtype=torch.bool, device=self.device)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.placed = torch.zeros_like(self.lifted)
        self.destination_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sampled_rows = torch.zeros_like(self.destination_ids)
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self._was_engaged = torch.zeros_like(self.lifted)
        self._arm_contacts = SceneEntityCfg("arm_contacts")
        self.full_start = torch.zeros_like(self.lifted)
        self.failed = torch.zeros_like(self.lifted)
        self.new_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waiting = torch.zeros_like(self.lifted)
        # A saved destination certifies a qualified release, independently of the next pickup.
        self._outfeed_destinations = torch.full_like(self.delivered, -1, dtype=torch.long)
        self._stable_steps = torch.zeros_like(self.delivered, dtype=torch.long)
        self._last_step = torch.full_like(self.new_success, -1)
        self._evaluation_step = -1
        self._bin_frames = torch.zeros((5, 3, 3), device=self.device)
        self._bin_frames[:, 0] = env.bin_directions
        self._bin_frames[:, 1, 0] = -env.bin_directions[:, 1]
        self._bin_frames[:, 1, 1] = env.bin_directions[:, 0]
        self._bin_frames[:, 2, 2] = 1.0
        self._bin_half_size = torch.tensor(BIN_INTERIOR_SIZE, device=self.device) / 2.0
        for name in (
            "success_rate",
            "correct_deliveries",
            "wrong_deliveries",
            "full_batch_success_rate",
            "controlled_placements",
            "invalid_releases",
            "arm_collisions",
            "missed_pickups",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired parcel-center position in the robot base frame [m], shape [num_envs, 3]."""
        robot = self._env.scene["robot"]
        root = robot.data.root_link_pose_w.torch
        position, _ = math_utils.subtract_frame_transforms(
            root[:, :3], root[:, 3:7], self.goal_positions + self._env.scene.env_origins
        )
        return position

    def record_release(self, release_mask: torch.Tensor) -> None:
        """Qualify the held parcel immediately before a policy command opens the vacuum."""
        env = self._env
        attempting = release_mask & env.gripper.engaged & self.lifted & ~self._waiting & ~self.placed
        ids = attempting.nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        state = _parcel_states(env)[ids, env.active_ids[ids]]
        local = state[:, :3] - env.scene.env_origins[ids]
        rotation = math_utils.matrix_from_quat(state[:, 3:7])
        half_size = env.parcel_sizes[env.active_ids[ids]] / 2.0
        frames = self._bin_frames[self.destination_ids[ids]]
        half_extents = (torch.bmm(frames, rotation).abs() * half_size[:, None]).sum(dim=-1)
        center = torch.bmm(frames, local[:, :, None]).squeeze(-1)
        bottom = local[:, 2] - (rotation[:, 2].abs() * half_size).sum(dim=-1)
        support_height = self.goal_positions[ids, 2] - half_size[:, 2]
        clearance = bottom - support_height
        supported = (
            (center[:, 0] - half_extents[:, 0] >= RECEIVING_PAD_RADIAL_BOUNDS[0])
            & (center[:, 0] + half_extents[:, 0] <= RECEIVING_PAD_RADIAL_BOUNDS[1])
            & (center[:, 1].abs() + half_extents[:, 1] <= RECEIVING_PAD_HALF_WIDTH)
        )
        belt_velocity = env.bin_directions[self.destination_ids[ids], :2] * env.cfg.conveyor_force.speed
        horizontal_speed = torch.minimum(
            torch.linalg.vector_norm(state[:, 7:9], dim=-1),
            torch.linalg.vector_norm(state[:, 7:9] - belt_velocity, dim=-1),
        )
        controlled = (
            supported
            & (
                torch.linalg.vector_norm(local[:, :2] - self.goal_positions[ids, :2], dim=-1)
                <= self.cfg.placement_xy_tolerance
            )
            & (clearance >= -0.003)
            & (clearance <= self.cfg.placement_max_height)
            & (horizontal_speed <= self.cfg.placement_max_horizontal_speed)
            & (state[:, 9].abs() <= self.cfg.placement_max_vertical_speed)
            & (torch.linalg.vector_norm(state[:, 10:13], dim=-1) <= self.cfg.placement_max_angular_speed)
            & (rotation[:, 2, 2] >= math.cos(self.cfg.placement_upright_tolerance))
            & torch.isfinite(state).all(dim=-1)
        )
        self.placed[ids] |= controlled
        self.failed[ids] |= ~controlled
        self.metrics["controlled_placements"][ids] += controlled.float()
        self.metrics["invalid_releases"][ids[~controlled]] = 1.0

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, torch.Tensor]:
        """Report only the resetting episodes; preserve all other environments' counters."""
        ids = self._ids[slice(None) if env_ids is None else env_ids]
        full = self.full_start[ids] & (self._env.episode_length_buf[ids] > 0)
        full_ids = ids[full]
        row_success = self.delivered[ids, self._reset_term.parcel_ids[self._sampled_rows[ids]]].float()
        full_success = row_success[full]
        full_batch = self.metrics["full_batch_success_rate"][full_ids]
        phase = self._reset_term.phase_ids[self._sampled_rows[ids]]
        phase_outcomes = {
            f"{name}_success_rate": row_success[(phase == index) & (self._env.episode_length_buf[ids] > 0)]
            for index, name in ((0, "place"), (1, "carry"), (2, "approach"), (4, "outfeed"))
        }
        extras = super().reset(ids)
        extras.update({name: outcomes for name, outcomes in phase_outcomes.items() if outcomes.numel()})
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
        """Verify all released parcels while the policy works on its current pickup."""
        env = self._env
        if self._evaluation_step == env.common_step_counter:
            return
        changed = self._last_step != env.episode_length_buf
        states = _parcel_states(env)
        local = states[:, :, :3] - env.scene.env_origins[:, None]
        self.placed[changed & env.gripper.engaged] = False
        self._waiting[changed & env.gripper.engaged] = False
        self._outfeed_destinations[
            self._ids[changed & env.gripper.engaged], env.active_ids[changed & env.gripper.engaged]
        ] = -1
        self.lifted |= (
            changed & ~self._waiting & env.gripper.engaged & (local[self._ids, env.active_ids, 2] > BELT_TOP_Z + 0.10)
        )
        released = changed & self.lifted & self.placed & ~env.gripper.engaged
        self._outfeed_destinations[self._ids[released], env.active_ids[released]] = self.destination_ids[released]
        pending = self._outfeed_destinations >= 0
        destinations = self._outfeed_destinations.clamp_min(0)
        offsets = local[:, :, None] - env.bin_positions
        in_bin_frame = torch.einsum("kij,npkj->npki", self._bin_frames, offsets)
        rotation = math_utils.matrix_from_quat(states[:, :, 3:7])
        half_extents = (
            torch.einsum("kij,npjl->npkil", self._bin_frames, rotation).abs()
            * (env.parcel_sizes[None, :, None, None] / 2.0)
        ).sum(dim=-1)
        inside = (in_bin_frame.abs() + half_extents < self._bin_half_size + 0.005).all(dim=-1)
        over_bin = (in_bin_frame.abs() < self._bin_half_size + 0.03).all(dim=-1)
        settled = torch.linalg.vector_norm(states[:, :, 7:10], dim=-1) < 0.35
        correct = inside.gather(-1, destinations[..., None]).squeeze(-1)
        eligible = pending & ~self.delivered & settled
        stable = eligible & correct
        self._stable_steps = torch.where(
            changed[:, None], torch.where(stable, self._stable_steps + 1, 0), self._stable_steps
        )
        completed = (self._stable_steps >= 3) & ~self.delivered
        wrong = eligible & inside.any(dim=-1) & ~correct
        fallen = (local[:, :, 2] < -0.65) & ~self.delivered
        correct_bin = over_bin.gather(-1, destinations[..., None]).squeeze(-1)
        fallen &= ~(pending & correct_bin)
        finite = torch.isfinite(states).all(dim=(1, 2)) & torch.isfinite(env.scene["robot"].data.joint_pos.torch).all(
            dim=-1
        )
        seal_broken = self._was_engaged & ~env.gripper.engaged & self.lifted & ~self.placed & ~self._waiting
        arm_contact = common_mdp.illegal_contact(
            env, threshold=self.cfg.arm_contact_force_threshold, sensor_cfg=self._arm_contacts
        )
        self.failed |= changed & (
            wrong.any(dim=-1) | fallen.any(dim=-1) | seal_broken | arm_contact | ~finite | invalid_policy_action(env)
        )
        self.metrics["arm_collisions"][changed & arm_contact] = 1.0
        completed &= changed[:, None] & ~self.failed[:, None]
        self.new_success = torch.where(changed, completed.sum(dim=-1), self.new_success)
        self.delivered |= completed
        # Only the first transfer starts from the sampled reset row. Later targets
        # still earn receipts and rewards, but must not relabel that row's outcome.
        initial = self._reset_term.parcel_ids[self._sampled_rows]
        first = completed[self._ids, initial].nonzero(as_tuple=False).flatten()
        self._reset_term.reset_sampler.success_monitor.success_update(
            self._reset_term.row_ids[first], completed[first, initial[first]]
        )
        env.episode_success |= self.new_success > 0
        count = self.delivered.sum(dim=-1)
        self.metrics["success_rate"][:] = env.episode_success.float()
        self.metrics["correct_deliveries"][:] = count.float()
        self.metrics["wrong_deliveries"][changed & wrong.any(dim=-1)] = 1.0
        self.metrics["full_batch_success_rate"][:] = (
            self.full_start & self.delivered.all(dim=-1) & ~self.failed
        ).float()
        self._last_step.copy_(env.episode_length_buf)
        self._was_engaged.copy_(env.gripper.engaged)
        self._evaluation_step = env.common_step_counter
        env.extras["successes"] = self.new_success > 0

    def _update_metrics(self) -> None:
        self.evaluate()

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        ids = self._ids[env_ids]
        phase = self._reset_term.phase_ids[self._reset_term.row_ids[ids]]
        self._sampled_rows[ids] = self._reset_term.row_ids[ids]
        self.delivered[ids] = False
        self.lifted[ids] = self._reset_term.lifted_rows[self._reset_term.row_ids[ids]]
        self.placed[ids] = phase == 4
        self._was_engaged[ids] = False
        self.destination_ids[ids] = self._reset_term.destination_ids[self._reset_term.row_ids[ids]]
        self.goal_positions[ids] = self._reset_term.goal_positions[self._reset_term.row_ids[ids]]
        self._outfeed_destinations[ids] = -1
        outfeed_ids = ids[phase == 4]
        self._outfeed_destinations[outfeed_ids, self._env.active_ids[outfeed_ids]] = self.destination_ids[outfeed_ids]
        self.full_start[ids] = phase == 3
        self.failed[ids] = self.new_success[ids] = self._waiting[ids] = False
        self._stable_steps[ids] = 0
        self._last_step[ids] = -1
        self._evaluation_step = -1

    def _update_command(self) -> None:
        env = self._env
        ids = (~env.gripper.engaged & ~self.failed & (env.episode_length_buf > 0)).nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        positions = torch.stack([parcel.data.root_pos_w.torch[ids] for parcel in env.parcels], dim=1)
        local = positions - env.scene.env_origins[ids, None]
        tool, _ = end_effector_pose(env)
        active_positions = positions[torch.arange(len(ids), device=self.device), env.active_ids[ids]]
        clearance = SUCTION_PAD_RADIUS + math.sqrt(2.0) * SUCTION_PAD_SPACING
        clear = self.placed[ids] & (
            torch.linalg.vector_norm(tool[ids] - active_positions, dim=-1)
            > (env.parcel_sizes[env.active_ids[ids]].norm(dim=-1) / 2.0 + clearance)
        )
        active_local = active_positions - env.scene.env_origins[ids]
        missed = (
            ~self.lifted[ids]
            & ~self.placed[ids]
            & (active_local[:, 0] > 0.85)
            & (active_local[:, 1] > INTAKE_CENTER[1])
        )
        self._waiting[ids] = clear | missed
        left = INTAKE_CENTER[0] - INTAKE_HALF_STRAIGHT
        # Reserve a parcel entering the turn, allowing time to return from the receiving pad.
        incoming = (
            (local[:, :, 0] > left - INTAKE_RADIUS - 0.05)
            & (local[:, :, 0] < left)
            & (local[:, :, 1] > INTAKE_CENTER[1] - INTAKE_RADIUS - 0.05)
            & (local[:, :, 1] < INTAKE_CENTER[1] - INTAKE_RADIUS / math.sqrt(2.0))
        )
        eligible = incoming & ~self.delivered[ids] & (self._outfeed_destinations[ids] < 0)
        anchor = torch.tensor((left, INTAKE_CENTER[1] - INTAKE_RADIUS), device=self.device)
        distances = torch.linalg.vector_norm(local[:, :, :2] - anchor, dim=-1)
        choice = distances.masked_fill(~eligible, torch.inf).argmin(dim=-1)
        available = (clear | missed) & eligible.any(dim=-1)
        selected = ids[available]
        self.metrics["missed_pickups"][selected] += missed[available].float()
        self._env.active_ids[selected] = choice[available]
        if self.cfg.randomize_destination:
            self.destination_ids[selected] = torch.randint(len(GOAL_POSITIONS), (len(selected),), device=self.device)
        else:
            self.destination_ids[selected] = self._env.parcel_classes[self._env.active_ids[selected]]
        offsets = torch.rand((len(selected), 2), device=self.device)
        for axis, bounds in enumerate((self.cfg.goal_radial_range, self.cfg.goal_lateral_range)):
            offsets[:, axis] = bounds[0] + offsets[:, axis] * (bounds[1] - bounds[0])
        self.goal_positions[selected] = self._reset_term._placement_goals(
            self._env.active_ids[selected], self.destination_ids[selected], offsets
        )
        self.lifted[selected] = self._waiting[selected] = False
        self.placed[selected] = self._was_engaged[selected] = False
        self._env.gripper.reset(selected)
        self._env.action_manager.reset(selected)


class ResetParcels(ManagerTermBase):
    """Sample the acquisition-to-receipt chain with shared outcome history."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        from .actions import solve_tool_ik

        super().__init__(cfg, env)
        # Sixteen neighboring states per path, at each of the two pickup anchors.
        samples = 16
        variants_per_phase = 2 * samples
        phase_count = 5
        rows = torch.arange(
            len(PARCEL_SIZES) * phase_count * variants_per_phase * len(GOAL_POSITIONS), device=self.device
        )
        self.row_count = rows.numel()
        self.destination_ids = rows % len(GOAL_POSITIONS)
        recipes = rows // len(GOAL_POSITIONS)
        self.parcel_ids = recipes // (phase_count * variants_per_phase)
        self.phase_ids = (recipes // variants_per_phase) % phase_count
        variants = recipes % variants_per_phase
        anchor = variants % 2
        fraction = (variants // 2).float() / (samples - 1)
        progress = (variants // 2).float() / samples
        self.reset_sampler = ResetStateSampler(cfg.params["sampling"], self.num_envs, self.row_count, self.device)
        self.row_ids = self.reset_sampler.row_ids
        self.held_rows = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._initial_poses = torch.stack(
            [env.scene[f"parcel_{index}"].data.default_root_pose.torch[0] for index in range(len(PARCEL_SIZES))]
        )
        self._arm_ids = env.scene["robot"].find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        self._parcel_sizes = torch.tensor(PARCEL_SIZES, device=self.device)
        sizes = self._parcel_sizes[self.parcel_ids]
        pick = torch.zeros((self.row_count, 3), device=self.device)
        pick[:, 0], pick[:, 1], pick[:, 2] = 0.54 + 0.10 * anchor, -0.19, BELT_TOP_Z + sizes[:, 2] + 0.004
        self._goal_centers = torch.tensor(GOAL_POSITIONS, device=self.device)
        self._goal_directions = torch.tensor(GOAL_DIRECTIONS, device=self.device)
        cfg = env.cfg.commands.sort
        offsets = torch.stack(
            [
                bounds[0] + fraction * (bounds[1] - bounds[0])
                for bounds in (cfg.goal_radial_range, cfg.goal_lateral_range)
            ],
            dim=-1,
        )
        self.goal_positions = self._placement_goals(self.parcel_ids, self.destination_ids, offsets)
        goal = self.goal_positions.clone()
        goal[:, 2] += sizes[:, 2] / 2.0 + 0.010
        home_arm = torch.tensor((0.0, -0.35, 0.0, -2.35, 0.0, 2.0, 0.0), device=self.device)
        home = franka_tool_position(home_arm[None], tool_offset_z=LINK7_TO_TOOL_Z)[0]
        pick_high, goal_high = pick.clone(), goal.clone()
        pick_high[:, 2] = goal_high[:, 2] = 0.30
        endpoint = torch.where((self.phase_ids == 1)[:, None], goal_high, home)
        lift_distance = pick_high[:, 2] - pick[:, 2]
        travel_distance = torch.linalg.vector_norm(endpoint - pick_high, dim=-1)
        distance = progress * (lift_distance + travel_distance)
        # Arc-length spacing joins contact to transport and contact to home without large gaps.
        path = pick + (pick_high - pick) * (distance / lift_distance).clamp(0.0, 1.0)[:, None]
        path += (endpoint - pick_high) * ((distance - lift_distance) / travel_distance).clamp(0.0, 1.0)[:, None]
        # Keep plentiful easy release seeds while filling the entire descent to transport height.
        descent = torch.lerp(goal, goal_high, fraction.pow(3)[:, None])
        self._tool_positions = torch.where((self.phase_ids == 0)[:, None], descent, path)
        self._tool_positions[self.phase_ids == 3] = home
        self._tool_positions[self.phase_ids == 4] = goal_high[self.phase_ids == 4]
        # Descent rows have already lifted. A newly sealed low lift must still clear the belt.
        self.lifted_rows = (
            (self.phase_ids == 0)
            | (self.phase_ids == 4)
            | ((self.phase_ids == 1) & (self._tool_positions[:, 2] - sizes[:, 2] / 2.0 - 0.002 > BELT_TOP_Z + 0.10))
        )
        self._arm_positions = home_arm.repeat(self.row_count, 1)
        partial = self.phase_ids != 3
        unique, inverse = torch.unique(self._tool_positions[partial], dim=0, return_inverse=True)
        self._arm_positions[partial] = solve_tool_ik(unique)[inverse]
        self._pick_positions = pick.clone()
        self._pick_positions[:, 2] -= 0.5 * sizes[:, 2] + 0.002
        # Earlier approach states need earlier parcel phases, including the moving left turn.
        left_x = INTAKE_CENTER[0] - INTAKE_HALF_STRAIGHT
        approach = torch.where(self.phase_ids == 3, 1.0, progress)
        distance = (pick[:, 0] - left_x) * (1.0 - approach) - math.pi * INTAKE_RADIUS / 2.0 * approach
        angle = (-distance / INTAKE_RADIUS).clamp_min(0.0)
        self._pick_positions[:, 0] = left_x + distance.clamp_min(0.0) - INTAKE_RADIUS * angle.sin()
        self._pick_positions[:, 1] = INTAKE_CENTER[1] + INTAKE_RADIUS * angle.cos()
        self._pick_yaw = angle
        self._parcel_velocities = torch.zeros((self.row_count, 6), device=self.device)
        speed = env.cfg.conveyor_force.speed
        unheld = (self.phase_ids == 2) | (self.phase_ids == 3)
        self._reserve_turn = unheld & (distance < -0.21)
        self._parcel_velocities[unheld, 0] = speed * angle[unheld].cos()
        self._parcel_velocities[unheld, 1] = speed * angle[unheld].sin()
        self._parcel_velocities[unheld & (distance < 0.0), 5] = -speed / INTAKE_RADIUS
        # Valid released predecessors teach the passive outfeed suffix, without seeding a receipt.
        outfeed = self.phase_ids == 4
        directions = self._goal_directions[self.destination_ids]
        radius = (self.goal_positions[:, :2] * directions[:, :2]).sum(dim=-1)
        last_supported = 1.95 - torch.linalg.vector_norm(sizes[:, :2], dim=-1) / 2.0 - 0.005
        travel = (last_supported - radius) * (1.0 - (1.0 - fraction).pow(3))
        self._pick_positions[outfeed] = self.goal_positions[outfeed]
        self._pick_positions[outfeed, :2] += directions[outfeed, :2] * travel[outfeed, None]
        self._pick_positions[outfeed, 2] += 0.002
        self._parcel_velocities[outfeed, :3] = directions[outfeed] * speed
        self._pick_yaw[outfeed] = 0.0

    def _placement_goals(
        self, parcel_ids: torch.Tensor, destination_ids: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Keep commanded parcel centers and their yaw-independent footprints on the pads."""
        sizes = self._parcel_sizes[parcel_ids]
        positions = self._goal_centers[destination_ids].clone()
        directions = self._goal_directions[destination_ids]
        half_diagonal = torch.linalg.vector_norm(sizes[:, :2], dim=-1) / 2.0
        radius = (positions[:, :2] * directions[:, :2]).sum(dim=-1)
        radial = offsets[:, 0].clamp(
            min=RECEIVING_PAD_RADIAL_BOUNDS[0] + half_diagonal + 0.005 - radius,
            max=RECEIVING_PAD_RADIAL_BOUNDS[1] - half_diagonal - 0.005 - radius,
        )
        lateral_limit = (RECEIVING_PAD_HALF_WIDTH - half_diagonal - 0.005).clamp_min(0.0)
        lateral = offsets[:, 1].clamp(min=-lateral_limit, max=lateral_limit)
        positions[:, :2] += directions[:, :2] * radial[:, None]
        positions[:, 0] -= directions[:, 1] * lateral
        positions[:, 1] += directions[:, 0] * lateral
        positions[:, 2] += sizes[:, 2] / 2.0
        return positions

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        sampling: ResetStateSamplerCfg,
        full_starts: bool = False,
    ) -> None:
        """Record pending failures before sampling and restoring selected environments."""
        del sampling
        ids = env.scene._ALL_INDICES[env_ids] if isinstance(env_ids, slice) else env_ids
        row_mask = self.phase_ids == 3 if full_starts else torch.ones_like(self.phase_ids, dtype=torch.bool)
        if not env.cfg.commands.sort.randomize_destination:
            row_mask &= self.destination_ids == torch.tensor(PARCEL_CLASS_IDS, device=self.device)[self.parcel_ids]
        sampled_received = env.command_manager.get_term("sort").delivered[ids, self.parcel_ids[self.row_ids[ids]]]
        rows = self.reset_sampler.reset(
            ids,
            sampled_received,
            valid=(env.episode_length_buf[ids] > 0) & ~sampled_received,
            row_mask=row_mask,
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
        # An upstream target frees slot zero; use it to clear the occupied left turn.
        positions[self._reserve_turn[rows, None] & (slots == len(env.parcels) - 2), :2] = self._initial_poses[0, :2]
        positions[:, :, 2] = BELT_TOP_Z + 0.5 * env.parcel_sizes[:, 2] + 0.002
        positions[:, :, 2] = torch.where(slots == len(env.parcels) - 1, self._initial_poses[-1, 2], positions[:, :, 2])
        active_position = self._pick_positions[rows].clone()
        held_position = self._tool_positions[rows].clone()
        held_position[:, 2] -= 0.5 * env.parcel_sizes[active, 2] + 0.002
        active_position = torch.where(self.held_rows[ids, None], held_position, active_position)
        positions[batch, active] = active_position
        yaw = (torch.rand((len(ids), len(env.parcels)), device=self.device) - 0.5) * 0.2
        yaw[batch, active] += self._pick_yaw[rows] * ~self.held_rows[ids]
        velocity = torch.zeros((len(ids), 6), device=self.device)
        parcel_velocities = torch.zeros((len(ids), len(env.parcels), 6), device=self.device)
        parcel_velocities[batch, active] = self._parcel_velocities[rows]
        for index, parcel in enumerate(env.parcels):
            pose = self._initial_poses[index].expand(len(ids), -1).clone()
            pose[:, :3] = positions[:, index]
            pose[:, 5], pose[:, 6] = torch.sin(0.5 * yaw[:, index]), torch.cos(0.5 * yaw[:, index])
            pose[:, :3] += env.scene.env_origins[ids]
            parcel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            parcel.write_root_velocity_to_sim_index(root_velocity=parcel_velocities[:, index], env_ids=ids)
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
