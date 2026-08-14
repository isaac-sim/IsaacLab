# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state sampling for conveyor transfer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

from ..conveyor_geometry import (
    BELT_CENTER_X,
    BELT_INNER_STRAIGHT_Y,
    BELT_OUTER_STRAIGHT_Y,
    BELT_TOP_Z,
    CUBE_INNER_SLOT_X,
    CUBE_OUTER_SLOT_X,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


CUBE_COUNT = 4
CUBE_SIZE = 0.04
CUBE_REST_Z = BELT_TOP_Z + 0.5 * CUBE_SIZE
TRANSFER_X = 0.52
LEFT_SIDE = 0
RIGHT_SIDE = 1
_BELT_CURRICULUM_FRACTIONS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0)
_GRASP_CLOSURE_FRACTIONS = (0.25, 0.50, 0.75, 0.90, 1.0)
_OPEN_FINGER_POSITION = 0.040
_CLOSED_FINGER_POSITION = 0.019
BELT_DEPLOYMENT_VARIANT = len(_BELT_CURRICULUM_FRACTIONS) - 1


class ConveyorResetRecipe(IntEnum):
    """Reset phases ordered from easiest to complete task start."""

    GOAL = 0
    PLACE = 1
    CARRY = 2
    LIFT = 3
    GRASP = 4
    PREGRASP = 5
    BELT = 6


@dataclass(frozen=True)
class ConveyorResetRow:
    """One physical reset state and transfer command."""

    recipe: ConveyorResetRecipe
    variant_id: int
    target_cube_id: int
    source_side_id: int
    arm_positions: tuple[float, ...]
    finger_position: float
    held: bool
    belt_range_fraction: float


_HOME_ARM = (0.0, -0.35, 0.0, -2.35, 0.0, 2.0, 0.78)

# Constrained IK anchors target x=0.52 m with a downward-facing tool. Side 0
# is the positive-y conveyor and side 1 is the negative-y conveyor.
_SOURCE_GRASP_ARM = (
    (0.5144946, 0.5427007, -0.0340481, -2.0731085, 0.0350549, 2.6153000, 1.2350098),
    (-0.3400387, 0.5475255, -0.1330232, -2.0714802, 0.1368175, 2.6109921, 0.2080209),
)
_SOURCE_PREGRASP_ARM = (
    (0.5190744, 0.3742786, -0.0406565, -2.0872751, 0.0236238, 2.4611441, 1.2428656),
    (-0.3220402, 0.3787346, -0.1588824, -2.0865274, 0.0928583, 2.4588233, 0.2380420),
)
_SOURCE_LIFT_ARM = (
    (0.5212608, 0.2372272, -0.0439391, -2.0514939, 0.0137043, 2.2884652, 1.2495379),
    (-0.3132103, 0.2404478, -0.1720021, -2.0512362, 0.0540920, 2.2875542, 0.2640870),
)
_TARGET_PLACE_ARM = (
    # Source left, target right.
    (-0.2921792, 0.4521663, -0.1853937, -2.0854943, 0.1398595, 2.5262992, 0.2062777),
    # Source right, target left.
    (0.4780173, 0.4442472, 0.0008774, -2.0873070, -0.0005178, 2.5315716, 1.2591928),
)
_CARRY_ARM = (0.0, 0.0137827, 0.0, -2.2661237, 0.0, 2.2798989, 0.78)


def _interpolate_arm(
    start: tuple[float, ...],
    end: tuple[float, ...],
    fractions: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    """Linearly interpolate validated joint-space anchors."""
    return tuple(
        tuple(
            start_value + fraction * (end_value - start_value)
            for start_value, end_value in zip(start, end, strict=True)
        )
        for fraction in fractions
    )


def _arm_position_variants(
    recipe: ConveyorResetRecipe,
    source_side_id: int,
) -> tuple[tuple[float, ...], ...]:
    """Return dense reset states along the nominal transfer trajectory."""
    source_grasp = _SOURCE_GRASP_ARM[source_side_id]
    source_pregrasp = _SOURCE_PREGRASP_ARM[source_side_id]
    source_lift = _SOURCE_LIFT_ARM[source_side_id]
    target_lift = _SOURCE_LIFT_ARM[1 - source_side_id]
    if recipe == ConveyorResetRecipe.GOAL:
        return (_SOURCE_PREGRASP_ARM[1 - source_side_id],)
    if recipe == ConveyorResetRecipe.PLACE:
        return _interpolate_arm(target_lift, _TARGET_PLACE_ARM[source_side_id], (0.25, 0.50, 0.75, 1.0))
    if recipe == ConveyorResetRecipe.CARRY:
        return (
            *_interpolate_arm(source_lift, _CARRY_ARM, (0.33, 0.66, 1.0)),
            *_interpolate_arm(_CARRY_ARM, target_lift, (0.33, 0.66, 1.0)),
        )
    if recipe == ConveyorResetRecipe.LIFT:
        return _interpolate_arm(source_grasp, source_lift, (0.25, 0.50, 0.75, 1.0))
    if recipe == ConveyorResetRecipe.GRASP:
        return (source_grasp,) * len(_GRASP_CLOSURE_FRACTIONS)
    if recipe == ConveyorResetRecipe.PREGRASP:
        # Include the exact open-gripper acquisition pose. The previous last
        # row stopped at 88% of the approach, leaving an approximately 1 cm
        # gap between reset-driven approach learning and the already-held
        # GRASP row. Dense near-contact rows make the physical close-and-lift
        # transition learnable from the same sparse delivery objective.
        return _interpolate_arm(source_pregrasp, source_grasp, (0.0, 0.50, 0.75, 0.92, 1.0))
    if recipe == ConveyorResetRecipe.BELT:
        return _interpolate_arm(source_pregrasp, _HOME_ARM, _BELT_CURRICULUM_FRACTIONS)
    raise ValueError(f"Unsupported reset recipe: {recipe}.")


def _finger_position_variants(recipe: ConveyorResetRecipe) -> tuple[float, ...]:
    """Return finger positions paired with a recipe's arm variants [m]."""
    variant_count = len(_arm_position_variants(recipe, LEFT_SIDE))
    if recipe == ConveyorResetRecipe.GRASP:
        return tuple(
            _OPEN_FINGER_POSITION + fraction * (_CLOSED_FINGER_POSITION - _OPEN_FINGER_POSITION)
            for fraction in _GRASP_CLOSURE_FRACTIONS
        )
    if recipe in (ConveyorResetRecipe.LIFT, ConveyorResetRecipe.CARRY, ConveyorResetRecipe.PLACE):
        return (_CLOSED_FINGER_POSITION,) * variant_count
    return (_OPEN_FINGER_POSITION,) * variant_count


def reset_variant_counts() -> tuple[int, ...]:
    """Return the number of trajectory variants in each reset recipe."""
    return tuple(len(_arm_position_variants(recipe, LEFT_SIDE)) for recipe in ConveyorResetRecipe)


def build_reset_rows() -> tuple[ConveyorResetRow, ...]:
    """Build the complete identity, direction, and dense-trajectory cross product."""
    return tuple(
        ConveyorResetRow(
            recipe=recipe,
            variant_id=variant_id,
            target_cube_id=cube_id,
            source_side_id=source_side,
            arm_positions=arm_positions,
            finger_position=finger_position,
            held=(
                recipe in (ConveyorResetRecipe.LIFT, ConveyorResetRecipe.CARRY, ConveyorResetRecipe.PLACE)
                or (recipe == ConveyorResetRecipe.GRASP and variant_id == len(_GRASP_CLOSURE_FRACTIONS) - 1)
            ),
            belt_range_fraction=_BELT_CURRICULUM_FRACTIONS[variant_id] if recipe == ConveyorResetRecipe.BELT else 0.0,
        )
        for recipe in ConveyorResetRecipe
        for cube_id in range(CUBE_COUNT)
        for source_side in (LEFT_SIDE, RIGHT_SIDE)
        for variant_id, (arm_positions, finger_position) in enumerate(
            zip(_arm_position_variants(recipe, source_side), _finger_position_variants(recipe), strict=True)
        )
    )


_FRANKA_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
)


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float, reference: torch.Tensor) -> torch.Tensor:
    """Return a fixed XYZ roll-pitch-yaw rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return reference.new_tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def franka_tool_position(joint_positions: torch.Tensor) -> torch.Tensor:
    """Compute Panda tool-center positions [m] from seven joints [rad]."""
    if joint_positions.shape[-1] != 7:
        raise ValueError("Franka reset forward kinematics expects seven joint positions.")
    shape = joint_positions.shape[:-1]
    joints = joint_positions.reshape(-1, 7)
    count = joints.shape[0]
    rotation = torch.eye(3, dtype=joints.dtype, device=joints.device).expand(count, -1, -1).clone()
    position = torch.zeros((count, 3), dtype=joints.dtype, device=joints.device)
    reference = joints[0] if count else joint_positions.new_zeros(7)
    for joint_id, (origin_position, origin_rpy) in enumerate(_FRANKA_JOINT_ORIGINS):
        origin = joint_positions.new_tensor(origin_position).expand(count, -1)
        position += torch.bmm(rotation, origin.unsqueeze(-1)).squeeze(-1)
        rotation = torch.matmul(rotation, _rotation_matrix_from_rpy(*origin_rpy, reference=reference))
        angle = joints[:, joint_id]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        zeros, ones = torch.zeros_like(angle), torch.ones_like(angle)
        joint_rotation = torch.stack(
            (
                torch.stack((cosine, -sine, zeros), dim=1),
                torch.stack((sine, cosine, zeros), dim=1),
                torch.stack((zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )
        rotation = torch.bmm(rotation, joint_rotation)
    tool_offset = joint_positions.new_tensor((0.0, 0.0, 0.2104)).expand(count, -1)
    position += torch.bmm(rotation, tool_offset.unsqueeze(-1)).squeeze(-1)
    return position.reshape(*shape, 3)


def side_inner_y(side_ids: torch.Tensor) -> torch.Tensor:
    """Return the reachable inner-straight y coordinate [m] for each side."""
    return torch.where(side_ids == LEFT_SIDE, BELT_INNER_STRAIGHT_Y, -BELT_INNER_STRAIGHT_Y)


def _balanced_cube_slots(
    target_cube_ids: torch.Tensor,
    source_side_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign one cube to every inner/outer racetrack run.

    Slots are ordered ``left inner``, ``left outer``, ``right inner``, and
    ``right outer``.  The commanded cube swaps with the canonical occupant of
    its source-side inner slot, keeping it reachable without duplicating or
    emptying any deployment slot.
    """
    if target_cube_ids.ndim != 1 or source_side_ids.shape != target_cube_ids.shape:
        raise ValueError("Target cube and source-side ids must be matching vectors.")
    if bool(torch.any((target_cube_ids < 0) | (target_cube_ids >= CUBE_COUNT))):
        raise ValueError("Target cube ids are out of range.")
    if bool(torch.any((source_side_ids != LEFT_SIDE) & (source_side_ids != RIGHT_SIDE))):
        raise ValueError("Source-side ids must be 0 (left) or 1 (right).")

    count = target_cube_ids.numel()
    cube_slots = torch.arange(CUBE_COUNT, device=target_cube_ids.device).expand(count, -1).clone()
    source_inner_slots = 2 * source_side_ids
    displaced_cube_ids = source_inner_slots
    target_original_slots = target_cube_ids
    cube_slots.scatter_(1, target_cube_ids.unsqueeze(1), source_inner_slots.unsqueeze(1))
    cube_slots.scatter_(1, displaced_cube_ids.unsqueeze(1), target_original_slots.unsqueeze(1))

    cube_sides = torch.div(cube_slots, 2, rounding_mode="floor")
    on_outer_run = torch.remainder(cube_slots, 2).bool()
    cube_x = torch.where(on_outer_run, CUBE_OUTER_SLOT_X, CUBE_INNER_SLOT_X)
    y_magnitude = torch.where(on_outer_run, BELT_OUTER_STRAIGHT_Y, BELT_INNER_STRAIGHT_Y)
    cube_y = torch.where(cube_sides == LEFT_SIDE, y_magnitude, -y_magnitude)
    return cube_slots, cube_sides, cube_x, cube_y


def _sample_collision_free_active_x(
    base_x: torch.Tensor,
    cube_sides: torch.Tensor,
    target_cube_ids: torch.Tensor,
    source_side_ids: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    minimum_separation: float = 0.055,
    attempts: int = 16,
) -> torch.Tensor:
    """Sample active-cube positions without overlapping an inactive cube."""
    if base_x.ndim != 2 or base_x.shape[1] != CUBE_COUNT or cube_sides.shape != base_x.shape:
        raise ValueError("base_x and cube_sides must have shape (N, CUBE_COUNT).")
    count = base_x.shape[0]
    expected_vector_shape = (count,)
    if any(value.shape != expected_vector_shape for value in (target_cube_ids, source_side_ids, lower, upper)):
        raise ValueError("Active-cube sampling controls must have shape (N,).")
    if minimum_separation <= 0.0 or attempts < 1:
        raise ValueError("Invalid collision-free active-cube sampling parameters.")

    cube_ids = torch.arange(CUBE_COUNT, device=base_x.device).expand(count, -1)
    inactive_on_source = (cube_sides == source_side_ids.unsqueeze(1)) & (cube_ids != target_cube_ids.unsqueeze(1))
    sampled = lower + torch.rand_like(lower) * (upper - lower)
    for _ in range(attempts):
        conflicts = torch.any(
            (torch.abs(sampled.unsqueeze(1) - base_x) < minimum_separation) & inactive_on_source, dim=1
        )
        replacement = lower + torch.rand_like(lower) * (upper - lower)
        sampled = torch.where(conflicts, replacement, sampled)

    conflicts = torch.any((torch.abs(sampled.unsqueeze(1) - base_x) < minimum_separation) & inactive_on_source, dim=1)
    fallback = torch.full_like(sampled, TRANSFER_X)
    return torch.where(conflicts, fallback, sampled)


class ConveyorResetStateTable(ManagerTermBase):
    """Restore validated transfer states spanning released goal to moving start."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._rows = build_reset_rows()
        self.recipe_ids = torch.tensor([row.recipe for row in self._rows], dtype=torch.long, device=env.device)
        self.target_cube_ids = torch.tensor(
            [row.target_cube_id for row in self._rows], dtype=torch.long, device=env.device
        )
        self.variant_ids = torch.tensor([row.variant_id for row in self._rows], dtype=torch.long, device=env.device)
        self.source_side_ids = torch.tensor(
            [row.source_side_id for row in self._rows], dtype=torch.long, device=env.device
        )
        self._arm_positions = torch.tensor(
            [row.arm_positions for row in self._rows], dtype=torch.float32, device=env.device
        )
        self._finger_positions = torch.tensor(
            [row.finger_position for row in self._rows], dtype=torch.float32, device=env.device
        )
        self.held_rows = torch.tensor([row.held for row in self._rows], dtype=torch.bool, device=env.device)
        self._belt_range_fractions = torch.tensor(
            [row.belt_range_fraction for row in self._rows], dtype=torch.float32, device=env.device
        )
        self._robot: Articulation = env.scene["robot"]
        self._cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
        self._arm_joint_ids = self._robot.find_joints("panda_joint[1-7]", preserve_order=True)[0]
        self._finger_joint_ids = self._robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)[0]
        if len(self._arm_joint_ids) != 7 or len(self._finger_joint_ids) != 2:
            raise ValueError("Conveyor transfer requires seven Panda arm joints and two finger joints.")
        self.row_ids = torch.randint(self.row_count, (env.num_envs,), dtype=torch.long, device=env.device)

    @property
    def row_count(self) -> int:
        """Number of immutable physical reset rows."""
        return len(self._rows)

    @property
    def recipe_names(self) -> tuple[str, ...]:
        """Stable reset recipe labels."""
        return tuple(recipe.name.lower() for recipe in ConveyorResetRecipe)

    def _filtered_rows(
        self,
        fixed_recipe: int | None,
        fixed_variant_id: int | None,
        fixed_target_cube_id: int | None,
        fixed_source_side_id: int | None,
    ) -> torch.Tensor:
        """Return rows matching optional deterministic evaluation controls."""
        mask = torch.ones(self.row_count, dtype=torch.bool, device=self.device)
        if fixed_recipe is not None:
            if not 0 <= fixed_recipe < len(ConveyorResetRecipe):
                raise ValueError(f"fixed_recipe must lie in [0, {len(ConveyorResetRecipe) - 1}].")
            mask &= self.recipe_ids == fixed_recipe
        if fixed_variant_id is not None:
            if fixed_variant_id < 0:
                raise ValueError("fixed_variant_id must be non-negative.")
            mask &= self.variant_ids == fixed_variant_id
        if fixed_target_cube_id is not None:
            if not 0 <= fixed_target_cube_id < CUBE_COUNT:
                raise ValueError(f"fixed_target_cube_id must lie in [0, {CUBE_COUNT - 1}].")
            mask &= self.target_cube_ids == fixed_target_cube_id
        if fixed_source_side_id is not None:
            if fixed_source_side_id not in (LEFT_SIDE, RIGHT_SIDE):
                raise ValueError("fixed_source_side_id must be 0 (left) or 1 (right).")
            mask &= self.source_side_ids == fixed_source_side_id
        return torch.nonzero(mask, as_tuple=False).flatten()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_recipe: int | None = None,
        fixed_variant_id: int | None = None,
        fixed_target_cube_id: int | None = None,
        fixed_source_side_id: int | None = None,
        belt_start_x_range: tuple[float, float] = (0.30, 0.82),
        cube_position_noise: float = 0.015,
        arm_joint_noise: float = 0.015,
    ) -> None:
        """Write sampled robot and four-cube states directly into simulation."""
        if env_ids is None or env_ids.numel() == 0:
            return
        if belt_start_x_range[0] >= belt_start_x_range[1]:
            raise ValueError("belt_start_x_range must be strictly increasing.")
        if cube_position_noise < 0.0 or arm_joint_noise < 0.0:
            raise ValueError("Reset randomization ranges must be non-negative.")

        if (
            fixed_recipe is None
            and fixed_variant_id is None
            and fixed_target_cube_id is None
            and fixed_source_side_id is None
        ):
            row_ids = self.row_ids[env_ids]
        else:
            candidates = self._filtered_rows(
                fixed_recipe,
                fixed_variant_id,
                fixed_target_cube_id,
                fixed_source_side_id,
            )
            if candidates.numel() == 0:
                raise RuntimeError("No conveyor reset rows match the fixed reset controls.")
            row_ids = candidates[torch.randint(candidates.numel(), (env_ids.numel(),), device=self.device)]
            self.row_ids[env_ids] = row_ids

        recipes = self.recipe_ids[row_ids]
        target_cube_ids = self.target_cube_ids[row_ids]
        source_side_ids = self.source_side_ids[row_ids]
        held_rows = self.held_rows[row_ids]

        arm_positions = self._arm_positions[row_ids].clone()
        if arm_joint_noise > 0.0:
            noise = (2.0 * torch.rand_like(arm_positions) - 1.0) * arm_joint_noise
            # Preserve the exact approach, closure, and held-object manifold.
            # Only deployment-like BELT starts receive robot randomization.
            noise[recipes != int(ConveyorResetRecipe.BELT)] = 0.0
            arm_positions += noise
        joint_positions = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_positions[:, self._arm_joint_ids] = arm_positions
        finger_positions = self._finger_positions[row_ids].to(dtype=joint_positions.dtype).unsqueeze(1).expand(-1, 2)
        joint_positions[:, self._finger_joint_ids] = finger_positions
        self._robot.set_joint_position_target_index(target=joint_positions, env_ids=env_ids)
        self._robot.set_joint_velocity_target_index(target=joint_velocities, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_positions, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_velocities, env_ids=env_ids)

        count = env_ids.numel()
        cube_slots, cube_sides, base_x, cube_y = _balanced_cube_slots(target_cube_ids, source_side_ids)
        base_x = base_x.to(dtype=arm_positions.dtype)
        cube_y = cube_y.to(dtype=arm_positions.dtype)
        if cube_position_noise > 0.0:
            base_x += (2.0 * torch.rand_like(base_x) - 1.0) * cube_position_noise

        active_lower = torch.full((count,), TRANSFER_X, dtype=arm_positions.dtype, device=self.device)
        active_upper = active_lower.clone()
        belt_rows = recipes == int(ConveyorResetRecipe.BELT)
        if bool(torch.any(belt_rows)):
            range_fraction = self._belt_range_fractions[row_ids]
            range_lower = TRANSFER_X + range_fraction * (belt_start_x_range[0] - TRANSFER_X)
            range_upper = TRANSFER_X + range_fraction * (belt_start_x_range[1] - TRANSFER_X)
            active_lower[belt_rows] = range_lower[belt_rows]
            active_upper[belt_rows] = range_upper[belt_rows]
        active_x = _sample_collision_free_active_x(
            base_x,
            cube_sides,
            target_cube_ids,
            source_side_ids,
            active_lower,
            active_upper,
        )

        # On a full deployment reset, mirror the other cube on the source
        # conveyor across the racetrack center.  Opposite straight runs then
        # differ by exactly half a lap even when the active start is sampled.
        source_outer_slots = 2 * source_side_ids + 1
        source_outer_cube = cube_slots == source_outer_slots.unsqueeze(1)
        mirrored_outer_x = (2.0 * BELT_CENTER_X - active_x).unsqueeze(1)
        base_x = torch.where(belt_rows.unsqueeze(1) & source_outer_cube, mirrored_outer_x, base_x)
        cube_positions = torch.stack(
            (base_x, cube_y, torch.full_like(base_x, CUBE_REST_Z)),
            dim=2,
        )

        active_positions = torch.stack(
            (active_x, side_inner_y(source_side_ids), torch.full_like(active_x, CUBE_REST_Z)),
            dim=1,
        )
        goal_rows = recipes == int(ConveyorResetRecipe.GOAL)
        active_positions[goal_rows, 1] = side_inner_y(1 - source_side_ids[goal_rows])
        if bool(torch.any(held_rows)):
            active_positions[held_rows] = franka_tool_position(arm_positions[held_rows])
        cube_positions.scatter_(
            1,
            target_cube_ids.view(-1, 1, 1).expand(-1, 1, 3),
            active_positions.unsqueeze(1),
        )

        identity_quaternion = arm_positions.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(count, -1)
        for cube_id, cube in enumerate(self._cubes):
            root_pose = cube.data.default_root_pose.torch[env_ids].clone()
            root_pose[:, :3] = cube_positions[:, cube_id] + env.scene.env_origins[env_ids]
            root_pose[:, 3:7] = identity_quaternion
            root_velocity = torch.zeros((count, 6), dtype=root_pose.dtype, device=self.device)
            cube.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            cube.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Keep the immutable reset table across environment resets."""


def select_next_transfer_cube(
    cube_positions: torch.Tensor,
    current_cube_ids: torch.Tensor,
    source_side_ids: torch.Tensor,
    transit_half_width: float = 0.14,
) -> torch.Tensor:
    """Sample the next numbered cube already located on each source belt.

    A different eligible cube is sampled uniformly. The just-placed cube is
    the fallback, so a valid goal remains available even when it is temporarily
    the only parcel on that conveyor.
    """
    if cube_positions.ndim != 3 or cube_positions.shape[1:] != (CUBE_COUNT, 3):
        raise ValueError(f"cube_positions must have shape (N, {CUBE_COUNT}, 3).")
    count = cube_positions.shape[0]
    if current_cube_ids.shape != (count,) or source_side_ids.shape != (count,):
        raise ValueError("Current cube and source-side ids must match the position batch.")
    if transit_half_width <= 0.0:
        raise ValueError("transit_half_width must be positive.")

    cube_ids = torch.arange(CUBE_COUNT, device=cube_positions.device).expand(count, -1)
    on_left = cube_positions[:, :, 1] > transit_half_width
    on_right = cube_positions[:, :, 1] < -transit_half_width
    candidates = torch.where(source_side_ids.unsqueeze(1) == LEFT_SIDE, on_left, on_right)
    alternatives = candidates & (cube_ids != current_cube_ids.unsqueeze(1))
    candidates = torch.where(torch.any(alternatives, dim=1, keepdim=True), alternatives, candidates)

    has_candidates = torch.any(candidates, dim=1)
    fallback = torch.zeros_like(candidates)
    fallback.scatter_(1, current_cube_ids.unsqueeze(1), True)
    candidates = torch.where(has_candidates.unsqueeze(1), candidates, fallback)
    return torch.multinomial(candidates.float(), 1).squeeze(1)
