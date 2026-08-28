# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-driven transfer commands for the conveyor Franka task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from ..conveyor_geometry import BELT_CENTER_X, BELT_HALF_STRAIGHT
from .kinematics import end_effector_pose
from .reset_events import CUBE_COUNT, ConveyorResetRecipe, select_next_transfer_cube, side_inner_y
from .rewards import current_transfer_potential, physical_cube_acquisition_mask

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def transfer_success_mask(
    cube_positions: torch.Tensor,
    cube_linear_velocities: torch.Tensor,
    tool_positions: torch.Tensor,
    finger_positions: torch.Tensor,
    target_side_ids: torch.Tensor,
    lateral_tolerance: float = 0.055,
    maximum_cube_speed: float = 0.65,
    minimum_finger_position: float = 0.027,
    minimum_tool_clearance: float = 0.055,
) -> torch.Tensor:
    """Return whether the active cube is released on its destination belt."""
    target_y = side_inner_y(target_side_ids)
    on_straight = torch.abs(cube_positions[:, 0] - BELT_CENTER_X) < BELT_HALF_STRAIGHT
    on_lane = torch.abs(cube_positions[:, 1] - target_y) < lateral_tolerance
    supported_height = (cube_positions[:, 2] > 0.045) & (cube_positions[:, 2] < 0.095)
    moving_safely = torch.linalg.vector_norm(cube_linear_velocities, dim=1) < maximum_cube_speed
    released = torch.amin(finger_positions, dim=1) > minimum_finger_position
    hand_clear = torch.linalg.vector_norm(tool_positions - cube_positions, dim=1) > minimum_tool_clearance
    return on_straight & on_lane & supported_height & moving_safely & released & hand_clear


class ConveyorTransferCommand(CommandTerm):
    """Command one numbered cube to the opposite belt and redraw on success."""

    cfg: ConveyorTransferCommandCfg

    def __init__(self, cfg: ConveyorTransferCommandCfg, env: ManagerBasedRLEnv) -> None:
        self._validate_cfg(cfg)
        super().__init__(cfg, env)

        reset_term = env.event_manager.get_term_cfg(cfg.reset_event_name).func
        required_reset_fields = ("row_ids", "recipe_ids", "target_cube_ids", "source_side_ids", "held_rows")
        if not all(hasattr(reset_term, name) for name in required_reset_fields):
            raise RuntimeError("ConveyorTransferCommand requires ConveyorResetStateTable reset metadata.")
        self._reset_term = reset_term
        self._robot: Articulation = env.scene["robot"]
        self._cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
        self._finger_joint_ids = self._robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)[0]

        self.target_cube_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.source_side_ids = torch.zeros_like(self.target_cube_ids)
        self.recipe_ids = torch.zeros_like(self.target_cube_ids)
        self.held_cube_ids = torch.full_like(self.target_cube_ids, -1)
        self.subgoal_start_steps = torch.zeros_like(self.target_cube_ids)
        self.transfer_counts = torch.zeros_like(self.target_cube_ids)
        self.direction_transfer_counts = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)

        self._stable_steps = torch.zeros_like(self.target_cube_ids)
        self.is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.new_success = torch.zeros_like(self.is_success)
        self.pending_success = torch.zeros_like(self.is_success)
        self.ever_success = torch.zeros_like(self.is_success)
        self.progress_ever_success = torch.zeros_like(self.is_success)
        self._target_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_evaluation_steps = torch.full_like(self.target_cube_ids, -1)
        self._resampling_from_reset = False

        self.metrics["success_rate"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.metrics["transfer_count"] = torch.zeros_like(self.metrics["success_rate"])
        self.metrics["left_to_right_transfers"] = torch.zeros_like(self.metrics["success_rate"])
        self.metrics["right_to_left_transfers"] = torch.zeros_like(self.metrics["success_rate"])

    @staticmethod
    def _validate_cfg(cfg: ConveyorTransferCommandCfg) -> None:
        """Validate success and reset-progress thresholds."""
        if cfg.minimum_subgoal_steps < 0 or cfg.hold_steps < 1:
            raise ValueError("minimum_subgoal_steps must be non-negative and hold_steps must be positive.")
        if cfg.lateral_tolerance <= 0.0 or cfg.maximum_cube_speed <= 0.0:
            raise ValueError("Conveyor placement tolerances and speed limits must be positive.")
        if cfg.minimum_finger_position <= 0.0 or cfg.minimum_tool_clearance <= 0.0:
            raise ValueError("Conveyor release thresholds must be positive.")
        if cfg.minimum_progress_steps < 0 or cfg.minimum_progress <= 0.0 or cfg.maximum_target_potential <= 0.0:
            raise ValueError("Conveyor reset-progress thresholds are invalid.")
        if (
            cfg.minimum_acquisition_lift <= 0.0
            or cfg.maximum_acquisition_tool_distance <= 0.0
            or cfg.maximum_acquisition_finger_position <= 0.0
        ):
            raise ValueError("Conveyor acquisition thresholds must be positive.")

    @property
    def command(self) -> torch.Tensor:
        """Return target-cube and destination-belt one-hot commands."""
        cube = torch.nn.functional.one_hot(self.target_cube_ids, num_classes=CUBE_COUNT)
        destination = torch.nn.functional.one_hot(1 - self.source_side_ids, num_classes=2)
        return torch.cat((cube, destination), dim=1).float()

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        """Log per-command completion and initialize commands from sampled reset rows."""
        ids = self._resolve_env_ids(env_ids)
        valid = self.command_counter[ids] > 0
        attempts = self.command_counter[ids].clamp_min(1)
        self.metrics["success_rate"][ids] = torch.where(
            valid,
            self.transfer_counts[ids].float() / attempts.float(),
            0.0,
        )
        self.metrics["transfer_count"][ids] = self.transfer_counts[ids].float()
        self.metrics["left_to_right_transfers"][ids] = self.direction_transfer_counts[ids, 0].float()
        self.metrics["right_to_left_transfers"][ids] = self.direction_transfer_counts[ids, 1].float()

        self._resampling_from_reset = True
        try:
            extras = super().reset(ids)
        finally:
            self._resampling_from_reset = False

        initial_potential = current_transfer_potential(self._env, command=self)
        self._target_potential[ids] = torch.clamp_max(
            initial_potential[ids] + self.cfg.minimum_progress,
            self.cfg.maximum_target_potential,
        )
        self._last_evaluation_steps[ids] = -1
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        return extras

    def evaluate(self) -> None:
        """Evaluate stable completion and reset-learning progress once per policy step."""
        evaluation_steps = self._env.episode_length_buf
        evaluate_mask = (self.command_counter > 0) & (self._last_evaluation_steps != evaluation_steps)
        if not bool(torch.any(evaluate_mask)):
            return

        positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in self._cubes), dim=1)
        velocities = torch.stack(tuple(cube.data.root_lin_vel_w.torch for cube in self._cubes), dim=1)
        index = self.target_cube_ids.view(self.num_envs, 1, 1).expand(-1, 1, 3)
        active_position = torch.gather(positions, 1, index).squeeze(1) - self._env.scene.env_origins
        active_velocity = torch.gather(velocities, 1, index).squeeze(1)
        tool_position, _ = end_effector_pose(self._env)
        tool_position = tool_position - self._env.scene.env_origins
        finger_positions = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        successful = transfer_success_mask(
            active_position,
            active_velocity,
            tool_position,
            finger_positions,
            1 - self.source_side_ids,
            lateral_tolerance=self.cfg.lateral_tolerance,
            maximum_cube_speed=self.cfg.maximum_cube_speed,
            minimum_finger_position=self.cfg.minimum_finger_position,
            minimum_tool_clearance=self.cfg.minimum_tool_clearance,
        )
        subgoal_steps = evaluation_steps - self.subgoal_start_steps
        successful &= subgoal_steps >= self.cfg.minimum_subgoal_steps

        next_stable_steps = torch.where(successful, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        self._stable_steps[evaluate_mask] = next_stable_steps[evaluate_mask]
        stable = self._stable_steps >= self.cfg.hold_steps
        new_success = evaluate_mask & stable & ~self.is_success & ~self.pending_success
        self.is_success[evaluate_mask] = stable[evaluate_mask]
        self.new_success.copy_(new_success)
        self.pending_success |= new_success
        self.ever_success |= new_success
        success_ids = new_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel():
            source_sides = self.source_side_ids[success_ids]
            self.transfer_counts[success_ids] += 1
            self.direction_transfer_counts[success_ids, source_sides] += 1

        potential = current_transfer_potential(self._env, command=self)
        progressed = (potential >= self._target_potential) & (evaluation_steps >= self.cfg.minimum_progress_steps)
        acquisition_recipe = (
            (self.recipe_ids == int(ConveyorResetRecipe.GRASP))
            | (self.recipe_ids == int(ConveyorResetRecipe.PREGRASP))
            | (self.recipe_ids == int(ConveyorResetRecipe.BELT))
        )
        physically_acquired = physical_cube_acquisition_mask(
            self._env,
            command=self,
            minimum_lift=self.cfg.minimum_acquisition_lift,
            maximum_tool_distance=self.cfg.maximum_acquisition_tool_distance,
            maximum_finger_position=self.cfg.maximum_acquisition_finger_position,
        )
        progressed &= ~acquisition_recipe | physically_acquired
        self.progress_ever_success |= evaluate_mask & progressed
        self._last_evaluation_steps[evaluate_mask] = evaluation_steps[evaluate_mask]
        self._env.extras["successes"] = self.ever_success.clone()

    def set_goal(self, target_cube_id: int, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Replace the active command using the selected cube's current conveyor."""
        if not isinstance(target_cube_id, int) or isinstance(target_cube_id, bool):
            raise TypeError("target_cube_id must be an integer.")
        if not 0 <= target_cube_id < CUBE_COUNT:
            raise ValueError(f"target_cube_id must lie in [0, {CUBE_COUNT - 1}].")
        ids = self._resolve_env_ids(env_ids)
        if ids.numel() == 0:
            return
        cube = self._cubes[target_cube_id]
        local_y = cube.data.root_pos_w.torch[ids, 1] - self._env.scene.env_origins[ids, 1]
        source_side_ids = (local_y < 0.0).long()
        target_cube_ids = torch.full_like(ids, target_cube_id)
        self._assign_goal(ids, target_cube_ids, source_side_ids)

    def _update_metrics(self) -> None:
        self.evaluate()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self._resolve_env_ids(env_ids)
        if ids.numel() == 0:
            return
        if self._resampling_from_reset:
            rows = self._reset_term.row_ids[ids]
            target_cube_ids = self._reset_term.target_cube_ids[rows]
            source_side_ids = self._reset_term.source_side_ids[rows]
            self.recipe_ids[ids] = self._reset_term.recipe_ids[rows]
            held_rows = self._reset_term.held_rows[rows]
            self.transfer_counts[ids] = 0
            self.direction_transfer_counts[ids] = 0
            self.ever_success[ids] = False
            self.progress_ever_success[ids] = False
            self._assign_goal(ids, target_cube_ids, source_side_ids)
            self.held_cube_ids[ids] = torch.where(held_rows, target_cube_ids, -1)
            self.subgoal_start_steps[ids] = 0
            return

        positions = torch.stack(tuple(cube.data.root_pos_w.torch[ids] for cube in self._cubes), dim=1)
        positions -= self._env.scene.env_origins[ids].unsqueeze(1)
        next_source_side_ids = 1 - self.source_side_ids[ids]
        next_cube_ids = select_next_transfer_cube(
            positions,
            self.target_cube_ids[ids],
            next_source_side_ids,
            transit_half_width=self.cfg.transit_half_width,
        )
        self._assign_goal(ids, next_cube_ids, next_source_side_ids)

    def _update_command(self) -> None:
        completed_ids = self.pending_success.nonzero(as_tuple=False).squeeze(-1)
        if completed_ids.numel():
            self._resample(completed_ids)

    def _assign_goal(
        self,
        env_ids: torch.Tensor,
        target_cube_ids: torch.Tensor,
        source_side_ids: torch.Tensor,
    ) -> None:
        """Publish one command and clear completion state from the previous command."""
        self.target_cube_ids[env_ids] = target_cube_ids
        self.source_side_ids[env_ids] = source_side_ids
        self.held_cube_ids[env_ids] = -1
        self.subgoal_start_steps[env_ids] = self._env.episode_length_buf[env_ids]
        self._stable_steps[env_ids] = 0
        self.is_success[env_ids] = False
        self.new_success[env_ids] = False
        self.pending_success[env_ids] = False
        self._last_evaluation_steps[env_ids] = -1

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        """Return validated environment indices on the command device."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if bool(torch.any((ids < 0) | (ids >= self.num_envs))):
            raise IndexError("Conveyor command environment indices are out of range.")
        return ids

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        raise NotImplementedError("Conveyor commands are visualized by the Newton goal selector.")

    def _debug_vis_callback(self, event) -> None:
        raise NotImplementedError("Conveyor commands are visualized by the Newton goal selector.")


@configclass
class ConveyorTransferCommandCfg(CommandTermCfg):
    """Configuration for success-driven, bidirectional cube-transfer commands."""

    class_type: type[ConveyorTransferCommand] | str = "{DIR}.commands:ConveyorTransferCommand"
    """Command-term implementation resolved lazily by the manager."""

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Time-based resampling interval [s]; success normally resamples first."""

    reset_event_name: str = "reset_from_state_table"
    """Reset event containing the immutable physical-state table."""

    transit_half_width: float = 0.14
    """Half-width of the corridor excluded when selecting a source-belt cube [m]."""

    minimum_subgoal_steps: int = 2
    """Minimum policy steps before a placement can succeed."""

    hold_steps: int = 3
    """Consecutive policy steps for which a placement must remain valid."""

    lateral_tolerance: float = 0.055
    """Maximum lateral error from the destination belt center [m]."""

    maximum_cube_speed: float = 0.65
    """Maximum cube speed for a completed placement [m/s]."""

    minimum_finger_position: float = 0.027
    """Minimum position of each finger for the cube to count as released [m]."""

    minimum_tool_clearance: float = 0.055
    """Minimum tool-to-cube distance for the cube to count as released [m]."""

    minimum_progress_steps: int = 3
    """Minimum policy steps before reset-learning progress can be credited."""

    minimum_progress: float = 0.35
    """Required increase in the dimensionless transfer potential after reset."""

    maximum_target_potential: float = 5.0
    """Upper bound for the dimensionless reset-progress target."""

    minimum_acquisition_lift: float = 0.025
    """Minimum cube lift used to validate physical acquisition [m]."""

    maximum_acquisition_tool_distance: float = 0.075
    """Maximum tool-to-cube distance used to validate physical acquisition [m]."""

    maximum_acquisition_finger_position: float = 0.030
    """Maximum finger position used to validate physical acquisition [m]."""
