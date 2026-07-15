# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Pour adapter for the reusable reset-dataset generation utilities.

This module owns proposal generation, Newton IK, rigid collision validation, objective scoring,
and validation of the task-specific direct-state contract used by the runtime reset curriculum.
Generic batching, integrity hashing, atomic persistence, and adaptive runtime sampling live in
``isaaclab_tasks.utils`` so another task only needs to provide its proposals and validators. Particles
are represented by the existing deterministic cup-local fill lattice; replay transforms that one
layout by the cached source-cup pose and starts the MPM solver with zero velocity, stress, and
history.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.utils.reset_dataset import (
    reset_dataset_collect_batches,
    reset_dataset_content_digest,
    reset_dataset_digest,
    reset_dataset_save_atomic,
    reset_dataset_validate_header,
)

if TYPE_CHECKING:
    from .pour_env import FrankaPourEnv


FRANKA_POUR_RESET_DATASET_FORMAT = "franka_pour_reset_dataset"
FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION = 6
FRANKA_POUR_RESET_DATASET_TASK_ID = "Isaac-Pour-Franka-Reset-Dataset-v0"
NON_GRASPING_CATEGORY = 0
GRASPING_CATEGORY = 1
RESET_DATASET_GRASPING_COUNT = 10_000
RESET_DATASET_NON_GRASPING_COUNT = 10_000
RESET_DATASET_NEAR_POUR_COUNT = 1_000

_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
_STATE_KEYS = (
    "arm_joint_position",
    "arm_joint_velocity",
    "finger_joint_position",
    "finger_joint_velocity",
    "finger_joint_target",
    "source_root_pose",
    "source_root_velocity",
    "target_root_pose",
    "target_root_velocity",
    "category",
    "objective",
    "objective_raw",
    "objective_components",
    "grasp_region",
    "grasp_side",
    "attempt_id",
    "particle_layout_id",
    "ik_cost",
    "ik_position_residual",
    "ik_rotation_residual",
)


@dataclass(frozen=True)
class FrankaPourResetDatasetGeneratorCfg:
    """Configuration for one statically valid reset-dataset candidate pool."""

    grasping_count: int = RESET_DATASET_GRASPING_COUNT
    non_grasping_count: int = RESET_DATASET_NON_GRASPING_COUNT
    batch_size: int = 256
    seed: int = 42
    max_attempt_multiplier: int = 100

    # Reserve an explicit, side-balanced near-goal stratum. These states are generated target-first
    # and still pass the same Newton IK, self-collision, obstacle, and support-surface rejection as
    # every broad grasping state.
    near_pour_grasp_count: int = RESET_DATASET_NEAR_POUR_COUNT
    near_pour_horizontal_radius: float = 0.02
    near_pour_height_range: tuple[float, float] = (0.15, 0.25)
    near_pour_tilt_angle_range: tuple[float, float] = (math.radians(120.0), math.radians(170.0))

    # A conservative cylindrical envelope around the Panda base.  Sampling the central 90% of
    # this envelope and rejecting through Newton IK defines the usable kinematic workspace without
    # relying on a hand-authored finite pose bank.
    workspace_central_fraction: float = 0.90
    workspace_radius_range: tuple[float, float] = (0.20, 0.82)
    workspace_azimuth_range: tuple[float, float] = (-math.pi, math.pi)
    workspace_height_range: tuple[float, float] = (0.08, 0.82)

    ik_seeds: int = 64
    ik_iterations: int = 160
    ik_noise_std: float = 0.75
    ik_max_cost: float = 1.0e-3
    ik_joint_margin: float = 0.015
    ik_max_position_residual: float = 0.003
    ik_max_rotation_residual: float = math.radians(3.0)
    ik_max_home_distance: float = 6.0

    # TCP-local +Y is the jaw axis.  Its very small standard deviation preserves a genuine
    # Gaussian without initializing the exact-width cup deeply inside either finger.
    grasp_position_std: tuple[float, float, float] = (0.0015, 0.00005, 0.0015)
    grasp_seating_max_offset: tuple[float, float, float] = (0.006, 0.00025, 0.006)
    grasp_seating_max_rotation_error: float = math.radians(1.0)
    non_grasping_min_tcp_source_distance: float = 0.12
    collision_penetration_tolerance: float = 1.0e-4
    finger_contact_penetration_tolerance: float = 0.00025
    obstacle_clearance: float = 0.001

    objective_distance_threshold: float = 0.15
    objective_target_horizontal_threshold: float = 0.15
    objective_target_height_threshold: float = 0.15
    objective_inversion_gate_horizontal_threshold: float = 0.07
    objective_weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    def __post_init__(self) -> None:
        """Reject invalid sampling contracts before allocating simulation state."""
        for field_name in ("grasping_count", "non_grasping_count", "batch_size", "max_attempt_multiplier"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if (
            not isinstance(self.near_pour_grasp_count, int)
            or isinstance(self.near_pour_grasp_count, bool)
            or not 0 <= self.near_pour_grasp_count <= self.grasping_count
        ):
            raise ValueError("near_pour_grasp_count must be an integer in [0, grasping_count].")
        if not 0.0 < self.workspace_central_fraction <= 1.0:
            raise ValueError("workspace_central_fraction must lie in (0, 1].")
        for field_name in ("workspace_radius_range", "workspace_azimuth_range", "workspace_height_range"):
            lower, upper = getattr(self, field_name)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"{field_name} must contain two finite increasing values.")
        for field_name in (
            "ik_noise_std",
            "ik_max_cost",
            "ik_joint_margin",
            "ik_max_position_residual",
            "ik_max_rotation_residual",
            "ik_max_home_distance",
            "non_grasping_min_tcp_source_distance",
            "grasp_seating_max_rotation_error",
            "finger_contact_penetration_tolerance",
            "near_pour_horizontal_radius",
            "objective_distance_threshold",
            "objective_target_horizontal_threshold",
            "objective_target_height_threshold",
            "objective_inversion_gate_horizontal_threshold",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if self.collision_penetration_tolerance < 0.0 or self.obstacle_clearance < 0.0:
            raise ValueError("Collision tolerances must be nonnegative.")
        for field_name in ("near_pour_height_range", "near_pour_tilt_angle_range"):
            lower, upper = getattr(self, field_name)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or lower >= upper:
                raise ValueError(f"{field_name} must contain two finite increasing positive values.")
        if self.near_pour_tilt_angle_range[1] >= math.pi:
            raise ValueError("near_pour_tilt_angle_range must remain below pi radians.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.grasp_position_std):
            raise ValueError("grasp_position_std must contain three finite nonnegative values.")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.grasp_seating_max_offset):
            raise ValueError("grasp_seating_max_offset must contain three finite positive values.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.objective_weights):
            raise ValueError("objective_weights must be finite and nonnegative.")
        if not math.isclose(sum(self.objective_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("objective_weights must sum to one.")
        if self.objective_inversion_gate_horizontal_threshold > self.objective_target_horizontal_threshold:
            raise ValueError("The inversion gate cannot be wider than the target-alignment kernel.")


def grasp_objective_components(
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    source_region_center: torch.Tensor | Sequence[float],
    cup_center_offset: torch.Tensor | Sequence[float],
    target_rim_height: float,
    distance_threshold: float = 0.15,
    target_horizontal_threshold: float = 0.15,
    target_height_threshold: float = 0.15,
    inversion_gate_horizontal_threshold: float | None = None,
) -> torch.Tensor:
    """Return distance, inversion, and target-alignment scores in ``[0, 1]``.

    Poses use environment-frame position followed by an XYZW quaternion.  The source asset root
    lies at the cup base, so ``cup_center_offset`` is rotated into the world before scoring.
    """
    if source_pose.ndim != 2 or source_pose.shape[1] != 7:
        raise ValueError(f"source_pose must have shape (N, 7), got {tuple(source_pose.shape)}.")
    if target_pose.shape != source_pose.shape:
        raise ValueError("target_pose must have the same (N, 7) shape as source_pose.")
    source_center = torch.as_tensor(source_region_center, device=source_pose.device, dtype=source_pose.dtype)
    center_offset_value = torch.as_tensor(cup_center_offset, device=source_pose.device, dtype=source_pose.dtype)
    if source_center.shape != (3,) or center_offset_value.shape != (3,):
        raise ValueError("source_region_center and cup_center_offset must each have shape (3,).")
    if inversion_gate_horizontal_threshold is None:
        inversion_gate_horizontal_threshold = target_horizontal_threshold
    thresholds = (
        distance_threshold,
        target_horizontal_threshold,
        target_height_threshold,
        inversion_gate_horizontal_threshold,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("Objective thresholds must be finite and positive.")

    count = source_pose.shape[0]
    center_offset = center_offset_value.expand(count, -1)
    cup_center = source_pose[:, :3] + math_utils.quat_apply(source_pose[:, 3:7], center_offset)
    distance_score = torch.linalg.vector_norm(cup_center - source_center, dim=-1).div(distance_threshold).clamp_(0, 1)

    local_up = torch.zeros((count, 3), device=source_pose.device, dtype=source_pose.dtype)
    local_up[:, 2] = 1.0
    world_up = math_utils.quat_apply(source_pose[:, 3:7], local_up)
    inversion_score = ((1.0 - world_up[:, 2]) * 0.5).clamp_(0, 1)

    horizontal_distance = torch.linalg.vector_norm(cup_center[:, :2] - target_pose[:, :2], dim=-1)
    horizontal_score = (1.0 - horizontal_distance / target_horizontal_threshold).clamp_(0, 1)
    target_rim_z = target_pose[:, 2] + float(target_rim_height)
    height_score = ((cup_center[:, 2] - target_rim_z) / target_height_threshold).clamp_(0, 1)
    target_score = horizontal_score * height_score
    # Inverting the cup away from the receiver is destructive, not task progress. Couple tilt
    # credit to the complete receiver-alignment score so it vanishes unless the cup is both above
    # and horizontally aligned with the bowl.
    centered_above_target = horizontal_distance <= inversion_gate_horizontal_threshold
    target_gated_inversion_score = inversion_score * target_score * centered_above_target
    return torch.stack((distance_score, target_gated_inversion_score, target_score), dim=-1)


def source_root_position_from_tcp_grasp(
    tcp_position: torch.Tensor,
    tcp_quaternion: torch.Tensor,
    source_quaternion: torch.Tensor,
    grasp_offset_source: torch.Tensor,
    seating_offset_tcp: torch.Tensor,
) -> torch.Tensor:
    """Place a source root so its configured grasp point is seated at the tool centre.

    ``seating_offset_tcp`` is the sampled Gaussian displacement of the cup grasp point from the
    tool centre, expressed in the TCP frame. Keeping this transform explicit prevents confusing
    the cup's geometric centre with the point along the fingers that is actually grasped.
    """
    expected_shape = tcp_position.shape
    if expected_shape[-1:] != (3,) or source_quaternion.shape != (*expected_shape[:-1], 4):
        raise ValueError("TCP positions and source quaternions must have matching (..., 3/4) shapes.")
    if tcp_quaternion.shape != source_quaternion.shape or seating_offset_tcp.shape != expected_shape:
        raise ValueError("TCP quaternions and seating offsets must match the source batch shape.")
    grasp_offset_source = torch.as_tensor(
        grasp_offset_source,
        device=tcp_position.device,
        dtype=tcp_position.dtype,
    )
    if grasp_offset_source.shape != (3,):
        raise ValueError("grasp_offset_source must have shape (3,).")
    grasp_position = tcp_position + math_utils.quat_apply(tcp_quaternion, seating_offset_tcp)
    return grasp_position - math_utils.quat_apply(
        source_quaternion,
        grasp_offset_source.expand_as(tcp_position),
    )


def above_target_tilted_mask(
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    cup_center_offset: torch.Tensor | Sequence[float],
    target_rim_height: float,
    max_horizontal_distance: float,
    min_vertical_clearance: float,
    min_tilt_angle: float,
) -> torch.Tensor:
    """Return states whose cup center is above the receiver and whose cup is sufficiently tilted."""
    if source_pose.ndim != 2 or source_pose.shape[1] != 7 or target_pose.shape != source_pose.shape:
        raise ValueError("source_pose and target_pose must have matching shape (N, 7).")
    thresholds = (target_rim_height, max_horizontal_distance, min_vertical_clearance, min_tilt_angle)
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("Near-pour geometry thresholds must be finite and positive.")
    if min_tilt_angle >= math.pi:
        raise ValueError("min_tilt_angle must remain below pi radians.")
    count = source_pose.shape[0]
    center_offset = torch.as_tensor(cup_center_offset, device=source_pose.device, dtype=source_pose.dtype)
    if center_offset.shape != (3,):
        raise ValueError("cup_center_offset must have shape (3,).")
    cup_center = source_pose[:, :3] + math_utils.quat_apply(source_pose[:, 3:7], center_offset.expand(count, -1))
    horizontal_distance = torch.linalg.vector_norm(cup_center[:, :2] - target_pose[:, :2], dim=-1)
    vertical_clearance = cup_center[:, 2] - (target_pose[:, 2] + float(target_rim_height))
    local_up = torch.zeros((count, 3), device=source_pose.device, dtype=source_pose.dtype)
    local_up[:, 2] = 1.0
    world_up = math_utils.quat_apply(source_pose[:, 3:7], local_up)
    return (
        (horizontal_distance <= max_horizontal_distance)
        & (vertical_clearance >= min_vertical_clearance)
        & (world_up[:, 2] <= math.cos(min_tilt_angle))
    )


def oriented_box_supported_by_bounds(
    pose: torch.Tensor,
    half_extents: torch.Tensor | Sequence[float],
    support_lower_xy: torch.Tensor | Sequence[float],
    support_upper_xy: torch.Tensor | Sequence[float],
    *,
    clearance: float = 0.0,
) -> torch.Tensor:
    """Return whether each oriented box footprint lies completely on a rectangular support."""
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError("pose must have shape (N, 7).")
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and nonnegative.")
    half = torch.as_tensor(half_extents, device=pose.device, dtype=pose.dtype)
    lower = torch.as_tensor(support_lower_xy, device=pose.device, dtype=pose.dtype)
    upper = torch.as_tensor(support_upper_xy, device=pose.device, dtype=pose.dtype)
    if half.shape != (3,) or lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("half_extents, support_lower_xy, and support_upper_xy must have shapes (3,), (2,), (2,).")
    if bool(torch.any(half <= 0.0)) or bool(torch.any(lower >= upper)):
        raise ValueError("Support bounds and half extents must define positive regions.")
    center_offset = torch.zeros((pose.shape[0], 3), device=pose.device, dtype=pose.dtype)
    center_offset[:, 2] = half[2]
    center = pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], center_offset)
    rotation = math_utils.matrix_from_quat(pose[:, 3:7]).abs()
    planar_radius = (rotation[:, :2, :] * half).sum(dim=-1) + clearance
    return ((center[:, :2] - planar_radius >= lower) & (center[:, :2] + planar_radius <= upper)).all(dim=-1)


def normalize_grasp_objectives(raw_objective: torch.Tensor) -> torch.Tensor:
    """Normalize one non-degenerate vector so its exact extrema are zero and one."""
    if raw_objective.ndim != 1 or raw_objective.numel() < 2:
        raise ValueError("raw_objective must be a one-dimensional tensor with at least two entries.")
    if not bool(torch.isfinite(raw_objective).all()):
        raise ValueError("raw_objective must contain only finite values.")
    minimum = raw_objective.min()
    span = raw_objective.max() - minimum
    if float(span) <= torch.finfo(raw_objective.dtype).eps:
        raise ValueError("Cannot normalize degenerate grasp objectives with zero range.")
    normalized = (raw_objective - minimum) / span
    # Avoid roundoff obscuring the cache contract's exact endpoints.
    normalized[torch.argmin(raw_objective)] = 0.0
    normalized[torch.argmax(raw_objective)] = 1.0
    return normalized


def oriented_boxes_overlap(
    center_a: torch.Tensor,
    quaternion_a: torch.Tensor,
    half_extents_a: Sequence[float],
    center_b: torch.Tensor,
    quaternion_b: torch.Tensor,
    half_extents_b: Sequence[float],
    *,
    clearance: float = 0.0,
) -> torch.Tensor:
    """Return batched OBB intersection using the complete 15-axis separating-axis test."""
    if center_a.ndim != 2 or center_a.shape[1] != 3 or center_b.shape != center_a.shape:
        raise ValueError("Both center tensors must have shape (N, 3).")
    count = center_a.shape[0]
    if quaternion_a.shape != (count, 4) or quaternion_b.shape != (count, 4):
        raise ValueError("Both quaternion tensors must have shape (N, 4).")
    if clearance < 0.0:
        raise ValueError("clearance must be nonnegative.")

    rotation_a = math_utils.matrix_from_quat(quaternion_a)
    rotation_b = math_utils.matrix_from_quat(quaternion_b)
    relative_rotation = rotation_a.transpose(-1, -2) @ rotation_b
    absolute_rotation = relative_rotation.abs() + 1.0e-6
    translation = (rotation_a.transpose(-1, -2) @ (center_b - center_a).unsqueeze(-1)).squeeze(-1)
    half_a = torch.as_tensor(half_extents_a, device=center_a.device, dtype=center_a.dtype) + clearance * 0.5
    half_b = torch.as_tensor(half_extents_b, device=center_a.device, dtype=center_a.dtype) + clearance * 0.5

    separated = torch.zeros(count, device=center_a.device, dtype=torch.bool)
    for axis in range(3):
        radius_b = (absolute_rotation[:, axis, :] * half_b).sum(dim=-1)
        separated |= translation[:, axis].abs() > half_a[axis] + radius_b
    for axis in range(3):
        projection = (translation * relative_rotation[:, :, axis]).sum(dim=-1).abs()
        radius_a = (absolute_rotation[:, :, axis] * half_a).sum(dim=-1)
        separated |= projection > radius_a + half_b[axis]
    for axis_a in range(3):
        for axis_b in range(3):
            other_a = (axis_a + 1) % 3
            last_a = (axis_a + 2) % 3
            other_b = (axis_b + 1) % 3
            last_b = (axis_b + 2) % 3
            projection = (
                translation[:, last_a] * relative_rotation[:, other_a, axis_b]
                - translation[:, other_a] * relative_rotation[:, last_a, axis_b]
            ).abs()
            radius_a = (
                half_a[other_a] * absolute_rotation[:, last_a, axis_b]
                + half_a[last_a] * absolute_rotation[:, other_a, axis_b]
            )
            radius_b = (
                half_b[other_b] * absolute_rotation[:, axis_a, last_b]
                + half_b[last_b] * absolute_rotation[:, axis_a, other_b]
            )
            separated |= projection > radius_a + radius_b
    return ~separated


def _derive_tabletop_support_bounds(
    env: FrankaPourEnv,
    source_env_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the transformed SeattleLab tabletop footprint from the live USD stage."""
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.cloner import resolve_clone_plan_source

    if source_env_path is None:
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = resolve_clone_plan_source(env._robot.cfg.prim_path, plan) if plan is not None else None
        if resolved is None:
            raise RuntimeError(f"Could not resolve clone-plan source for {env._robot.cfg.prim_path!r}.")
        source_env_path = resolved[0]

    stage = sim_utils.get_current_stage()
    collision_path = f"{source_env_path.rstrip('/')}/Table/Collisions/Cube"
    collision_prim = stage.GetPrimAtPath(collision_path)
    if not collision_prim.IsValid():
        raise RuntimeError(f"Could not find the SeattleLab tabletop collision prim at {collision_path!r}.")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    world_range = bbox_cache.ComputeWorldBound(collision_prim).ComputeAlignedRange()
    lower_w = torch.tensor(tuple(world_range.GetMin()), device=env.device, dtype=torch.float32)
    upper_w = torch.tensor(tuple(world_range.GetMax()), device=env.device, dtype=torch.float32)
    lower = lower_w[:2] - env.env_origins[0, :2]
    upper = upper_w[:2] - env.env_origins[0, :2]
    if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all() & torch.all(lower < upper)):
        raise RuntimeError("Could not derive finite positive SeattleLab tabletop support bounds.")
    return lower, upper


def build_franka_pour_reset_task_contract(env: FrankaPourEnv) -> dict[str, Any]:
    """Build the canonical physics, geometry, and frame contract for reset-dataset compatibility."""
    import newton
    import warp as wp

    task_cfg = env.cfg
    source_half = torch.as_tensor(task_cfg.cup_grasp_box_half, dtype=torch.float32)
    target_half = torch.tensor(
        (
            task_cfg.target_cup_inner_width * 0.5 + task_cfg.target_cup_wall_thickness,
            task_cfg.target_cup_inner_depth * 0.5 + task_cfg.target_cup_wall_thickness,
            (task_cfg.target_cup_cavity_depth + task_cfg.target_cup_bottom_thickness) * 0.5,
        ),
        dtype=torch.float32,
    )
    cup_center_offset = torch.tensor((0.0, 0.0, float(source_half[2])), dtype=torch.float32)
    source_region_center = torch.as_tensor(task_cfg.cup_reset_pos, dtype=torch.float32) + cup_center_offset
    arm_limits = env._joint_pos_limits_t[0, env._arm_joint_ids].detach().cpu()
    finger_limits = env._joint_pos_limits_t[0, env._finger_joint_ids].detach().cpu()
    finger_position_range = (float(finger_limits[:, 0].max()), float(finger_limits[:, 1].min()))
    tabletop_lower, tabletop_upper = _derive_tabletop_support_bounds(env)
    material = task_cfg.media_material
    particle_spacing = float(task_cfg.voxel_size) / float(task_cfg.particles_per_cell)
    robot_spawn = task_cfg.scene.robot.spawn
    table_spawn = task_cfg.scene.table.spawn
    return {
        "robot_asset": str(getattr(robot_spawn, "usd_path", type(robot_spawn).__name__)),
        "table_asset": str(getattr(table_spawn, "usd_path", type(table_spawn).__name__)),
        "newton_version": str(getattr(newton, "__version__", "unknown")),
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "source_box_half": tuple(float(value) for value in source_half),
        "target_box_half": tuple(float(value) for value in target_half),
        "target_rim_height": float(target_half[2] * 2.0),
        "source_mesh_vertices_sha256": reset_dataset_digest(torch.as_tensor(env._cup_vertices)),
        "source_mesh_indices_sha256": reset_dataset_digest(torch.as_tensor(env._cup_indices)),
        "target_mesh_vertices_sha256": reset_dataset_digest(torch.as_tensor(env._target_vertices)),
        "target_mesh_indices_sha256": reset_dataset_digest(torch.as_tensor(env._target_indices)),
        "tabletop_support_lower_xy": tuple(float(value) for value in tabletop_lower),
        "tabletop_support_upper_xy": tuple(float(value) for value in tabletop_upper),
        "cup_grasp_tcp_quat_c": tuple(float(value) for value in task_cfg.cup_grasp_tcp_quat_c),
        "cup_grasp_height": float(task_cfg.cup_grasp_height),
        "source_region_center": tuple(float(value) for value in source_region_center),
        "source_radius_range": task_cfg.curriculum_randomized_source_radius_range,
        "source_position_range": tuple(task_cfg.curriculum_randomized_source_position_range),
        "source_azimuth_range": float(task_cfg.curriculum_randomized_source_azimuth_range),
        "target_center_xy": tuple(task_cfg.curriculum_randomized_target_center_xy),
        "target_position_range": tuple(task_cfg.curriculum_randomized_target_position_range),
        "tcp_body_name": task_cfg.tcp_body_name,
        "tcp_offset_pos": tuple(float(value) for value in task_cfg.tcp_offset_pos),
        "tcp_offset_rot": tuple(float(value) for value in task_cfg.tcp_offset_rot),
        "arm_home": tuple(float(value) for value in task_cfg.arm_home),
        "arm_joint_limits": arm_limits,
        "gripper_position_range": finger_position_range,
        "gripper_open_pos": float(task_cfg.gripper_open_pos),
        "gripper_preload_pos": float(task_cfg.gripper_preload_pos),
        "gripper_grasp_reset_target": float(task_cfg.actions.gripper_action.close_position),
        "gripper_contact_min_deflection": float(task_cfg.actions.gripper_action.contact_min_deflection),
        "cup_mass": float(task_cfg.cup_mass),
        "source_cup_friction": float(task_cfg.source_cup_friction),
        "target_cup_friction": float(task_cfg.target_cup_friction),
        "cup_grasp_box_friction": float(task_cfg.cup_grasp_box_friction),
        "grasp_contact_ke": float(task_cfg.grasp_contact_ke),
        "grasp_contact_kd": float(task_cfg.grasp_contact_kd),
        "grasp_contact_kf": float(task_cfg.grasp_contact_kf),
        "collider_margin": float(task_cfg.collider_margin),
        "simulation_dt": float(task_cfg.sim.dt),
        "gravity": tuple(float(value) for value in task_cfg.sim.gravity),
        "policy_decimation": int(task_cfg.decimation),
        "physics_substeps": int(task_cfg.physics_substeps),
        "rigid_entry_substeps": int(task_cfg.rigid_entry_substeps),
        "mpm_entry_substeps": int(task_cfg.mpm_entry_substeps),
        "mpm_iterations": int(task_cfg.mpm_iterations),
        "proxy_iterations": int(task_cfg.proxy_iterations),
        "proxy_mass_scale": float(task_cfg.proxy_mass_scale),
        "particle_workspace_lower_bound": tuple(task_cfg.particle_workspace_lower_bound),
        "particle_workspace_upper_bound": tuple(task_cfg.particle_workspace_upper_bound),
        "particle_max_velocity": float(task_cfg.particle_max_velocity),
        "particle_count": int(env._media_local_points_t.shape[0]),
        "particle_spacing": particle_spacing,
        "particle_mass": particle_spacing**3 * float(material.density),
        "particle_radius": 0.5 * particle_spacing,
        "media_fill_fraction": float(task_cfg.media_fill_frac),
        "media_material": {
            name: float(getattr(material, name))
            for name in (
                "density",
                "young_modulus",
                "poisson_ratio",
                "viscosity",
                "friction",
                "damping",
                "yield_pressure",
                "tensile_yield_ratio",
                "yield_stress",
                "hardening",
                "dilatancy",
            )
        },
        "particle_layout_sha256": reset_dataset_digest(env._media_local_points_t),
    }


def reset_dataset_content_sha256(payload: Mapping[str, Any]) -> str:
    """Return the content hash while excluding the hash field itself."""
    return reset_dataset_content_digest(payload)


def build_reset_dataset_payload(
    states: Mapping[str, torch.Tensor],
    particle_local_positions: torch.Tensor,
    metadata: Mapping[str, Any],
    cfg: FrankaPourResetDatasetGeneratorCfg,
) -> dict[str, Any]:
    """Build, hash, and validate a CPU-only direct-state cache payload."""
    missing = sorted(set(_STATE_KEYS) - set(states))
    extra = sorted(set(states) - set(_STATE_KEYS))
    if missing or extra:
        raise ValueError(f"State tensor keys differ from the schema: missing={missing}, extra={extra}.")
    cpu_states = {key: value.detach().cpu().contiguous() for key, value in states.items()}
    local_positions = particle_local_positions.detach().cpu().to(dtype=torch.float32).contiguous()
    if local_positions.ndim == 3 and local_positions.shape[0] == 1:
        local_positions = local_positions[0]
    if local_positions.ndim != 2 or local_positions.shape[1] != 3 or local_positions.shape[0] == 0:
        raise ValueError("particle_local_positions must have shape (P, 3) or (1, P, 3), with P > 0.")
    particle_layouts = {
        "local_position": local_positions.unsqueeze(0),
        "local_velocity": torch.zeros_like(local_positions).unsqueeze(0),
    }
    sampler_cfg = asdict(cfg)
    task_contract = metadata.get("task_contract", {})
    contract_sha256 = reset_dataset_digest({"sampler_cfg": sampler_cfg, "task_contract": task_contract})
    payload: dict[str, Any] = {
        "schema_version": FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
        "format": FRANKA_POUR_RESET_DATASET_FORMAT,
        "contract_sha256": contract_sha256,
        "metadata": {
            **dict(metadata),
            "sampler_cfg": sampler_cfg,
            "state_count": cfg.grasping_count + cfg.non_grasping_count,
            "category_names": ("non_grasping", "grasping"),
            "category_counts": torch.tensor((cfg.non_grasping_count, cfg.grasping_count), dtype=torch.int64),
            "joint_names": _ARM_JOINT_NAMES + _FINGER_JOINT_NAMES,
            "frame": "environment",
            "quaternion_order": "xyzw",
            "particle_solver_state": "fresh_zero",
        },
        "states": cpu_states,
        "particle_layouts": particle_layouts,
    }
    payload["content_sha256"] = reset_dataset_content_sha256(payload)
    validate_reset_dataset(
        payload,
        expected_grasping_count=cfg.grasping_count,
        expected_non_grasping_count=cfg.non_grasping_count,
    )
    return payload


def validate_reset_dataset(
    payload: Mapping[str, Any],
    *,
    expected_grasping_count: int | None = None,
    expected_non_grasping_count: int | None = None,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate schema, state invariants, category quotas, and the complete content hash."""
    metadata, sampler_cfg, task_contract, states = _validate_cache_header(payload)
    state_count = _validate_state_tensor_schema(states)
    grasping, non_grasping = _validate_category_counts(
        states,
        metadata,
        sampler_cfg,
        state_count,
        expected_grasping_count,
        expected_non_grasping_count,
    )
    _validate_objective_values(states, metadata, grasping, non_grasping)
    _validate_state_invariants(states, sampler_cfg, task_contract, grasping, non_grasping)
    _validate_particle_layouts(payload)
    if expected_task_contract is not None:
        stored_contract_digest = reset_dataset_digest(task_contract)
        expected_contract_digest = reset_dataset_digest(expected_task_contract)
        if stored_contract_digest != expected_contract_digest:
            raise ValueError(
                "Reset dataset task contract does not match the current environment: "
                f"{stored_contract_digest} != {expected_contract_digest}. Regenerate candidates "
                "with the same task configuration used for validation and training."
            )
    expected_hash = payload.get("content_sha256")
    if not isinstance(expected_hash, str) or expected_hash != reset_dataset_content_sha256(payload):
        raise ValueError("Reset dataset content hash does not match its payload.")


def validate_production_reset_dataset(
    payload: Mapping[str, Any],
    *,
    expected_grasping_count: int = RESET_DATASET_GRASPING_COUNT,
    expected_non_grasping_count: int = RESET_DATASET_NON_GRASPING_COUNT,
    expected_task_contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate a reset dataset and its dynamic-simulation provenance.

    Candidate datasets intentionally satisfy the state schema so the offline validator can replay
    them. Production training additionally requires the provenance marker written only after every
    retained row has passed dynamic validation in the real task.

    Args:
        payload: Loaded Franka Pour reset-dataset payload.
        expected_grasping_count: Required number of grasping rows.
        expected_non_grasping_count: Required number of non-grasping rows.
        expected_task_contract: Optional canonical current-task contract to compare in full.

    Raises:
        ValueError: If the payload is not a valid reset dataset or lacks valid dynamic-validation
            provenance.
    """
    validate_reset_dataset(
        payload,
        expected_grasping_count=expected_grasping_count,
        expected_non_grasping_count=expected_non_grasping_count,
        expected_task_contract=expected_task_contract,
    )
    metadata = payload["metadata"]
    marker = metadata.get("dynamic_validation")
    if not isinstance(marker, Mapping):
        raise ValueError(
            "Production reset datasets require a dynamic_validation metadata marker; "
            "run validate_franka_pour_reset_dataset.py on the candidate dataset first."
        )

    source_hash = marker.get("source_content_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("dynamic_validation.source_content_sha256 must be a lowercase SHA-256.")
    if source_hash == payload["content_sha256"]:
        raise ValueError("Dynamic-validation source and production content hashes must differ.")

    integer_fields = {
        "steps": 1,
        "settle_steps": 0,
        "failure_dwell_steps": 1,
        "balance_trimmed": 0,
    }
    for name, minimum in integer_fields.items():
        value = marker.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"dynamic_validation.{name} must be an integer >= {minimum}.")
    if marker["settle_steps"] >= marker["steps"]:
        raise ValueError("dynamic_validation.settle_steps must be smaller than steps.")

    failure_counts = marker.get("failure_counts")
    if not isinstance(failure_counts, Mapping) or not failure_counts:
        raise ValueError("dynamic_validation.failure_counts must be a nonempty mapping.")
    for name, count in failure_counts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dynamic_validation.failure_counts keys must be nonempty strings.")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("dynamic_validation.failure_counts values must be nonnegative integers.")


def select_production_reset_rows(
    states: Mapping[str, torch.Tensor],
    dynamically_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the exact balanced production quotas from an oversampled validated pool."""
    category = states["category"]
    region = states["grasp_region"]
    side = states["grasp_side"]
    objective = states["objective"]
    if dynamically_valid.shape != category.shape or dynamically_valid.dtype != torch.bool:
        raise ValueError("dynamically_valid must be a Boolean vector aligned with reset rows.")

    keep = torch.zeros_like(dynamically_valid)
    non_grasping_rows = torch.nonzero(
        dynamically_valid & (category == NON_GRASPING_CATEGORY),
        as_tuple=False,
    ).flatten()
    if non_grasping_rows.numel() < RESET_DATASET_NON_GRASPING_COUNT:
        raise RuntimeError(
            "Dynamic validation retained only "
            f"{non_grasping_rows.numel()}/{RESET_DATASET_NON_GRASPING_COUNT} required non-grasping states. "
            "Generate a larger candidate pool or improve proposal validity."
        )
    keep[non_grasping_rows[:RESET_DATASET_NON_GRASPING_COUNT]] = True

    required_per_side = {
        0: (RESET_DATASET_GRASPING_COUNT - RESET_DATASET_NEAR_POUR_COUNT) // 4,
        1: RESET_DATASET_NEAR_POUR_COUNT // 4,
    }
    for region_id, required in required_per_side.items():
        for side_id in range(4):
            rows = torch.nonzero(
                dynamically_valid & (category == GRASPING_CATEGORY) & (region == region_id) & (side == side_id),
                as_tuple=False,
            ).flatten()
            rows = rows[torch.argsort(objective[rows], descending=True, stable=True)]
            if rows.numel() < required:
                label = "near-pour" if region_id == 1 else "broad"
                raise RuntimeError(
                    f"Dynamic validation retained only {rows.numel()}/{required} required "
                    f"{label} grasping states for side {side_id}. Generate a larger candidate "
                    "pool or improve proposal validity."
                )
            keep[rows[:required]] = True

    if int(keep.sum()) != RESET_DATASET_GRASPING_COUNT + RESET_DATASET_NON_GRASPING_COUNT:
        raise RuntimeError("Balanced validation did not produce the exact 20,000-state production quota.")
    return keep, dynamically_valid & ~keep


def _validate_cache_header(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, torch.Tensor]]:
    """Validate and return the typed top-level cache mappings."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Reset-dataset metadata must be a mapping.")
    sampler_cfg = metadata.get("sampler_cfg")
    task_contract = metadata.get("task_contract", {})
    if not isinstance(sampler_cfg, Mapping) or not isinstance(task_contract, Mapping):
        raise ValueError("Reset-dataset sampling and task contracts must be mappings.")
    metadata, states = reset_dataset_validate_header(
        payload,
        expected_format=FRANKA_POUR_RESET_DATASET_FORMAT,
        expected_schema_version=FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
        expected_contract={"sampler_cfg": sampler_cfg, "task_contract": task_contract},
    )
    if not isinstance(states, Mapping) or set(states) != set(_STATE_KEYS):
        raise ValueError("Reset dataset has invalid state tensor keys.")
    if not all(isinstance(value, torch.Tensor) for value in states.values()):
        raise TypeError("Every reset-dataset state field must be a tensor.")
    return metadata, sampler_cfg, task_contract, states


def _validate_state_tensor_schema(states: Mapping[str, torch.Tensor]) -> int:
    """Validate all tensor shapes, scalar dtypes, and finite floating-point values."""
    state_count = int(states["category"].numel())
    expected_shapes = {
        "arm_joint_position": (state_count, 7),
        "arm_joint_velocity": (state_count, 7),
        "finger_joint_position": (state_count, 2),
        "finger_joint_velocity": (state_count, 2),
        "finger_joint_target": (state_count, 2),
        "source_root_pose": (state_count, 7),
        "source_root_velocity": (state_count, 6),
        "target_root_pose": (state_count, 7),
        "target_root_velocity": (state_count, 6),
        "category": (state_count,),
        "objective": (state_count,),
        "objective_raw": (state_count,),
        "objective_components": (state_count, 3),
        "grasp_region": (state_count,),
        "grasp_side": (state_count,),
        "attempt_id": (state_count,),
        "particle_layout_id": (state_count,),
        "ik_cost": (state_count,),
        "ik_position_residual": (state_count,),
        "ik_rotation_residual": (state_count,),
    }
    for key, shape in expected_shapes.items():
        if tuple(states[key].shape) != shape:
            raise ValueError(f"State field {key!r} must have shape {shape}, got {tuple(states[key].shape)}.")
    expected_dtypes = {
        "category": torch.int8,
        "grasp_region": torch.int8,
        "grasp_side": torch.int8,
        "attempt_id": torch.int64,
        "particle_layout_id": torch.int32,
    }
    for key, dtype in expected_dtypes.items():
        if states[key].dtype != dtype:
            raise ValueError(f"State field {key!r} must use dtype {dtype}, got {states[key].dtype}.")
    floating_keys = [key for key in _STATE_KEYS if states[key].is_floating_point()]
    if any(not bool(torch.isfinite(states[key]).all()) for key in floating_keys):
        raise ValueError("Reset dataset contains a non-finite state value.")
    return state_count


def _validate_category_counts(
    states: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    sampler_cfg: Mapping[str, Any],
    state_count: int,
    expected_grasping_count: int | None,
    expected_non_grasping_count: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate category encoding, metadata census, and requested exact quotas."""
    category = states["category"]
    grasping = category == GRASPING_CATEGORY
    non_grasping = category == NON_GRASPING_CATEGORY
    if not bool((grasping | non_grasping).all()):
        raise ValueError("State categories must be zero (non-grasping) or one (grasping).")
    grasping_count = int(grasping.sum())
    non_grasping_count = int(non_grasping.sum())
    metadata_counts = metadata.get("category_counts")
    if not isinstance(metadata_counts, torch.Tensor) or not torch.equal(
        metadata_counts.cpu(), torch.tensor((non_grasping_count, grasping_count), dtype=torch.int64)
    ):
        raise ValueError("Reset-dataset metadata category counts do not match its states.")
    if metadata.get("state_count") != state_count:
        raise ValueError("Reset-dataset metadata state count does not match its states.")
    if expected_grasping_count is None:
        expected_grasping_count = int(sampler_cfg["grasping_count"])
    if expected_non_grasping_count is None:
        expected_non_grasping_count = int(sampler_cfg["non_grasping_count"])
    if expected_grasping_count is not None and grasping_count != expected_grasping_count:
        raise ValueError(f"Expected {expected_grasping_count} grasping states, got {grasping_count}.")
    if expected_non_grasping_count is not None and non_grasping_count != expected_non_grasping_count:
        raise ValueError(f"Expected {expected_non_grasping_count} non-grasping states, got {non_grasping_count}.")
    return grasping, non_grasping


def _validate_objective_values(
    states: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    grasping: torch.Tensor,
    non_grasping: torch.Tensor,
) -> None:
    """Validate fixed non-grasp values and globally normalized grasp objectives."""
    grasping_count = int(grasping.sum())
    non_grasping_count = int(non_grasping.sum())
    objective = states["objective"]
    if non_grasping_count and not bool((objective[non_grasping] == -1.0).all()):
        raise ValueError("Every non-grasping objective must be exactly -1.")
    if non_grasping_count and (
        not bool((states["objective_raw"][non_grasping] == -1.0).all())
        or not bool((states["objective_components"][non_grasping] == -1.0).all())
    ):
        raise ValueError("Every non-grasping raw objective and component must be exactly -1.")
    if grasping_count:
        grasp_objective = objective[grasping]
        if not bool(((grasp_objective >= 0.0) & (grasp_objective <= 1.0)).all()):
            raise ValueError("Every grasping objective must lie in [0, 1].")
        if grasping_count > 1 and (float(grasp_objective.min()) != 0.0 or float(grasp_objective.max()) != 1.0):
            raise ValueError("Grasping objectives must include exact normalized extrema zero and one.")
        grasp_sides = states["grasp_side"][grasping]
        if not bool(((grasp_sides >= 0) & (grasp_sides <= 3)).all()):
            raise ValueError("Grasping states must use side IDs in [0, 3].")
        grasp_regions = states["grasp_region"][grasping]
        if not bool(((grasp_regions == 0) | (grasp_regions == 1)).all()):
            raise ValueError("Grasping states must use broad region zero or near-pour region one.")
        near_pour_count = int((grasp_regions == 1).sum())
        expected_near_pour_count = int(metadata["sampler_cfg"]["near_pour_grasp_count"])
        if near_pour_count != expected_near_pour_count:
            raise ValueError(f"Expected {expected_near_pour_count} near-pour grasping states, got {near_pour_count}.")
        components = states["objective_components"][grasping]
        if not bool(((components >= 0.0) & (components <= 1.0)).all()):
            raise ValueError("Every grasping objective component must lie in [0, 1].")
        weights = metadata.get("objective_weights")
        if not isinstance(weights, torch.Tensor) or weights.shape != (3,):
            raise ValueError("Reset-dataset metadata must contain three objective weights.")
        if metadata.get("objective_component_names") != (
            "source_distance",
            "target_gated_inversion",
            "target_alignment",
        ):
            raise ValueError("Reset-dataset metadata has unsupported objective-component semantics.")
        expected_raw = (components * weights.to(dtype=components.dtype)).sum(dim=-1)
        if not bool(torch.allclose(states["objective_raw"][grasping], expected_raw, atol=1.0e-6, rtol=0.0)):
            raise ValueError("Grasping raw objectives do not match their weighted components.")
        expected_normalized = normalize_grasp_objectives(states["objective_raw"][grasping])
        if not bool(torch.allclose(grasp_objective, expected_normalized, atol=1.0e-6, rtol=0.0)):
            raise ValueError("Grasping objectives do not match global min/max normalization.")
        raw_min_max = metadata.get("objective_raw_min_max")
        actual_min_max = torch.stack(
            (states["objective_raw"][grasping].min(), states["objective_raw"][grasping].max())
        ).cpu()
        if not isinstance(raw_min_max, torch.Tensor) or not bool(
            torch.allclose(raw_min_max.cpu(), actual_min_max, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("Reset-dataset metadata raw-objective extrema do not match its states.")
        if grasping_count % 4 == 0:
            side_counts = torch.bincount(grasp_sides.to(torch.long), minlength=4)
            if not bool((side_counts == grasping_count // 4).all()):
                raise ValueError("Grasping states must be exactly balanced over all four horizontal sides.")
        if expected_near_pour_count and expected_near_pour_count % 4 == 0:
            near_pour_side_counts = torch.bincount(grasp_sides[grasp_regions == 1].to(torch.long), minlength=4)
            if not bool((near_pour_side_counts == expected_near_pour_count // 4).all()):
                raise ValueError("Near-pour grasping states must be exactly balanced over all four sides.")
    if non_grasping_count:
        if not bool((states["grasp_side"][non_grasping] == -1).all()):
            raise ValueError("Non-grasping states must use grasp_side=-1.")
        if not bool((states["grasp_region"][non_grasping] == -1).all()):
            raise ValueError("Non-grasping states must use grasp_region=-1.")


def _validate_state_invariants(
    states: Mapping[str, torch.Tensor],
    sampler_cfg: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    grasping: torch.Tensor,
    non_grasping: torch.Tensor,
) -> None:
    """Validate gripper, velocity, pose, attempt, and particle-layout invariants."""
    state_count = states["category"].shape[0]
    grasping_count = int(grasping.sum())
    non_grasping_count = int(non_grasping.sum())
    source_box_half = task_contract.get("source_box_half") if isinstance(task_contract, Mapping) else None
    if source_box_half is not None and grasping_count:
        exact_finger_position = float(source_box_half[1])
        if not bool((states["finger_joint_position"][grasping] == exact_finger_position).all()):
            raise ValueError("Grasping finger positions must exactly match the source-cup width.")
        grasp_reset_target = task_contract.get("gripper_grasp_reset_target")
        contact_min_deflection = task_contract.get("gripper_contact_min_deflection")
        cup_grasp_height = task_contract.get("cup_grasp_height")
        if grasp_reset_target is None or contact_min_deflection is None or cup_grasp_height is None:
            raise ValueError("The reset-cache task contract must define grasp placement and drive preload.")
        grasp_reset_target = float(grasp_reset_target)
        contact_min_deflection = float(contact_min_deflection)
        if not math.isfinite(float(cup_grasp_height)) or float(cup_grasp_height) <= 0.0:
            raise ValueError("The configured cup grasp height must be finite and positive.")
        if not 0.0 <= grasp_reset_target < exact_finger_position:
            raise ValueError("The grasp reset target must lie inside the geometric source-cup contact position.")
        if exact_finger_position - grasp_reset_target < contact_min_deflection:
            raise ValueError("The grasp reset target does not retain the required finger-contact deflection.")
        if not bool((states["finger_joint_target"][grasping] == grasp_reset_target).all()):
            raise ValueError("Grasping finger targets must equal the configured reset target.")
        near_pour = grasping & (states["grasp_region"] == 1)
        if bool(near_pour.any()):
            target_rim_height = float(task_contract["target_rim_height"])
            near_pour_valid = above_target_tilted_mask(
                states["source_root_pose"][near_pour],
                states["target_root_pose"][near_pour],
                cup_center_offset=(0.0, 0.0, float(source_box_half[2])),
                target_rim_height=target_rim_height,
                max_horizontal_distance=float(sampler_cfg["near_pour_horizontal_radius"]),
                min_vertical_clearance=float(sampler_cfg["near_pour_height_range"][0]),
                min_tilt_angle=float(sampler_cfg["near_pour_tilt_angle_range"][0]),
            )
            if not bool(near_pour_valid.all()):
                raise ValueError("A tagged near-pour grasp is not geometrically above and tilted over the target.")
    gripper_position_range = task_contract.get("gripper_position_range") if isinstance(task_contract, Mapping) else None
    if gripper_position_range is not None and non_grasping_count:
        lower, upper = (float(value) for value in gripper_position_range)
        non_grasp_fingers = states["finger_joint_position"][non_grasping]
        if not bool(((non_grasp_fingers >= lower) & (non_grasp_fingers <= upper)).all()):
            raise ValueError("Non-grasping finger positions must lie in the complete valid opening range.")
        if not bool(torch.allclose(non_grasp_fingers[:, 0], non_grasp_fingers[:, 1], atol=0.0, rtol=0.0)):
            raise ValueError("The two Franka fingers must be restored symmetrically.")
    if torch.unique(states["attempt_id"]).numel() != state_count:
        raise ValueError("Accepted reset states must have unique attempt IDs.")
    if non_grasping_count and not bool(
        torch.allclose(
            states["finger_joint_target"][non_grasping],
            states["finger_joint_position"][non_grasping],
            atol=0.0,
            rtol=0.0,
        )
    ):
        raise ValueError("Non-grasping finger targets must preserve the sampled opening.")
    if not bool((states["particle_layout_id"] == 0).all()):
        raise ValueError("The current cache schema requires the shared particle layout ID zero.")
    for velocity_key in (
        "arm_joint_velocity",
        "finger_joint_velocity",
        "source_root_velocity",
        "target_root_velocity",
    ):
        if not bool((states[velocity_key] == 0.0).all()):
            raise ValueError(f"Reset-dataset field {velocity_key!r} must start at zero.")
    for pose_key in ("source_root_pose", "target_root_pose"):
        norms = torch.linalg.vector_norm(states[pose_key][:, 3:7], dim=-1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-4, rtol=0.0)):
            raise ValueError(f"{pose_key} contains a non-unit quaternion.")
    tabletop_lower_xy = task_contract.get("tabletop_support_lower_xy")
    tabletop_upper_xy = task_contract.get("tabletop_support_upper_xy")
    target_box_half = task_contract.get("target_box_half")
    obstacle_clearance = float(sampler_cfg.get("obstacle_clearance", 0.0))
    if tabletop_lower_xy is not None and tabletop_upper_xy is not None and target_box_half is not None:
        target_supported = oriented_box_supported_by_bounds(
            states["target_root_pose"],
            target_box_half,
            tabletop_lower_xy,
            tabletop_upper_xy,
            clearance=obstacle_clearance,
        )
        if not bool(target_supported.all()):
            raise ValueError("A target bowl reset pose is not fully supported by the tabletop.")
        if non_grasping_count and source_box_half is not None:
            source_supported = oriented_box_supported_by_bounds(
                states["source_root_pose"][non_grasping],
                source_box_half,
                tabletop_lower_xy,
                tabletop_upper_xy,
                clearance=obstacle_clearance,
            )
            if not bool(source_supported.all()):
                raise ValueError("A non-grasping source-cup reset pose is not fully supported by the tabletop.")


def _validate_particle_layouts(payload: Mapping[str, Any]) -> None:
    """Validate the one shared deterministic local particle layout."""
    layouts = payload.get("particle_layouts")
    if not isinstance(layouts, Mapping) or set(layouts) != {"local_position", "local_velocity"}:
        raise ValueError("Reset dataset has invalid particle layouts.")
    local_position = layouts["local_position"]
    local_velocity = layouts["local_velocity"]
    if (
        not isinstance(local_position, torch.Tensor)
        or local_position.ndim != 3
        or local_position.shape[0] != 1
        or local_position.shape[2] != 3
        or local_velocity.shape != local_position.shape
        or not bool(torch.isfinite(local_position).all())
        or not bool((local_velocity == 0.0).all())
    ):
        raise ValueError("Reset-dataset particle layout must be one finite (P, 3) layout with zero velocity.")


def save_reset_dataset(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Atomically write a validated reset dataset."""
    reset_dataset_save_atomic(payload, output_path, validator=validate_reset_dataset)


class FrankaPourResetDatasetGenerator:
    """Generate the complete Franka Pour reset dataset through rejection sampling."""

    def __init__(self, env: FrankaPourEnv, cfg: FrankaPourResetDatasetGeneratorCfg):
        if cfg.near_pour_grasp_count in (0, cfg.grasping_count):
            raise ValueError("Candidate generation requires both broad and near-pour grasping states.")
        if cfg.near_pour_grasp_count % 4 != 0:
            raise ValueError("near_pour_grasp_count must be divisible by four for side-balanced proposals.")
        if (cfg.grasping_count - cfg.near_pour_grasp_count) % 4 != 0:
            raise ValueError("The broad grasping count must be divisible by four for side-balanced proposals.")
        self.env = env
        self.task_cfg = env.cfg
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(cfg.seed)
        self._next_attempt_id = 0
        self._attempt_counts = {NON_GRASPING_CATEGORY: 0, GRASPING_CATEGORY: 0}
        self._rejection_counts: dict[int, dict[str, int]] = {
            NON_GRASPING_CATEGORY: defaultdict(int),
            GRASPING_CATEGORY: defaultdict(int),
        }
        self._build_ik_context()

    @torch.inference_mode()
    def generate(self) -> dict[str, Any]:
        """Generate, score, validate, and return the configured exact category quotas."""
        near_pour_grasping = self._sample_category(
            GRASPING_CATEGORY,
            self.cfg.near_pour_grasp_count,
            near_pour=True,
        )
        broad_grasping = self._sample_category(
            GRASPING_CATEGORY,
            self.cfg.grasping_count - self.cfg.near_pour_grasp_count,
            near_pour=False,
        )
        grasping = {key: torch.cat((near_pour_grasping[key], broad_grasping[key]), dim=0) for key in _STATE_KEYS}
        raw = grasping["objective_raw"]
        grasping["objective"] = normalize_grasp_objectives(raw)
        non_grasping = self._sample_category(NON_GRASPING_CATEGORY, self.cfg.non_grasping_count)
        states = {key: torch.cat((grasping[key], non_grasping[key]), dim=0) for key in _STATE_KEYS}
        permutation = torch.randperm(
            self.cfg.grasping_count + self.cfg.non_grasping_count,
            device=self.device,
            generator=self.generator,
        )
        states = {key: value[permutation] for key, value in states.items()}

        metadata = {
            "seed": self.cfg.seed,
            "storage_order": "seeded_random_permutation",
            "source_region_center": self._source_region_center.detach().cpu(),
            "cup_center_offset": self._cup_center_offset.detach().cpu(),
            "objective_weights": torch.tensor(self.cfg.objective_weights, dtype=torch.float32),
            "objective_component_names": (
                "source_distance",
                "target_gated_inversion",
                "target_alignment",
            ),
            "objective_raw_min_max": torch.stack((raw.min(), raw.max())).detach().cpu(),
            "attempt_counts": torch.tensor(
                (self._attempt_counts[NON_GRASPING_CATEGORY], self._attempt_counts[GRASPING_CATEGORY]),
                dtype=torch.int64,
            ),
            "rejection_counts": {
                name: torch.tensor(
                    (
                        self._rejection_counts[NON_GRASPING_CATEGORY].get(name, 0),
                        self._rejection_counts[GRASPING_CATEGORY].get(name, 0),
                    ),
                    dtype=torch.int64,
                )
                for name in sorted(
                    set(self._rejection_counts[NON_GRASPING_CATEGORY]) | set(self._rejection_counts[GRASPING_CATEGORY])
                )
            },
            "task_contract": self._task_contract(),
        }
        return build_reset_dataset_payload(states, self.env._media_local_points_t, metadata, self.cfg)

    def _build_ik_context(self) -> None:
        import newton
        import warp as wp
        from isaaclab_newton.cloner import copy_newton_source_builder
        from isaaclab_newton.ik.newton_ik_objectives_cfg import (
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
        )
        from isaaclab_newton.ik.newton_ik_solver import NewtonIKSolver
        from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

        import isaaclab.sim as sim_utils
        from isaaclab.cloner import resolve_clone_plan_source

        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = resolve_clone_plan_source(self.env._robot.cfg.prim_path, plan) if plan is not None else None
        if resolved is None:
            raise RuntimeError(f"Could not resolve clone-plan source for {self.env._robot.cfg.prim_path!r}.")
        source_builder = copy_newton_source_builder(resolved[0])
        prototype_origin = -self.env.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*prototype_origin.tolist()), wp.quat_identity())
        self._prototype_builder = newton.ModelBuilder(up_axis=source_builder.up_axis)
        self._prototype_builder.add_builder(source_builder, xform=prototype_xform)
        if not any(
            "/Table/" in str(label) or str(label).endswith("/Table") for label in self._prototype_builder.shape_label
        ):
            table_prim_path = self.env.scene["table"].cfg.prim_path
            table_resolved = resolve_clone_plan_source(table_prim_path, plan)
            if table_resolved is None:
                raise RuntimeError(f"Could not resolve clone-plan source for {table_prim_path!r}.")
            table_builder = copy_newton_source_builder(table_resolved[0])
            self._prototype_builder.add_builder(table_builder, xform=prototype_xform)
        if not any(
            "/Table/" in str(label) or str(label).endswith("/Table") for label in self._prototype_builder.shape_label
        ):
            raise RuntimeError("Reset-dataset validation could not import SeattleLab table collision geometry.")
        self._tabletop_support_lower_xy, self._tabletop_support_upper_xy = _derive_tabletop_support_bounds(
            self.env, resolved[0]
        )
        self._ik_model = self._prototype_builder.finalize(device=str(self.device))

        body_names = [str(label).rsplit("/", 1)[-1] for label in self._ik_model.body_label]
        hand_matches = [index for index, name in enumerate(body_names) if name == self.task_cfg.tcp_body_name]
        if len(hand_matches) != 1:
            raise RuntimeError(f"Expected one IK body named {self.task_cfg.tcp_body_name!r}, got {hand_matches}.")
        self._hand_id = hand_matches[0]
        joint_names = [str(label).rsplit("/", 1)[-1] for label in self._ik_model.joint_label]
        joint_q_start = wp.to_torch(self._ik_model.joint_q_start).to(device=self.device, dtype=torch.long)

        def coordinate_id(name: str) -> int:
            matches = [index for index, joint_name in enumerate(joint_names) if joint_name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one IK joint named {name!r}, got {matches}.")
            return int(joint_q_start[matches[0]].item())

        self._arm_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in _ARM_JOINT_NAMES], device=self.device, dtype=torch.long
        )
        self._finger_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in _FINGER_JOINT_NAMES], device=self.device, dtype=torch.long
        )
        self._arm_limits = self.env._joint_pos_limits_t[0, self.env._arm_joint_ids].to(self.device)
        finger_limits = self.env._joint_pos_limits_t[0, self.env._finger_joint_ids].to(self.device)
        self._finger_position_range = (float(finger_limits[:, 0].max()), float(finger_limits[:, 1].min()))
        if self._finger_position_range[0] >= self._finger_position_range[1]:
            raise RuntimeError(f"The two Franka fingers have no shared valid opening range: {finger_limits}.")
        self._arm_home = torch.tensor(self.task_cfg.arm_home, device=self.device, dtype=torch.float32)
        self._tcp_offset_pos = torch.tensor(self.task_cfg.tcp_offset_pos, device=self.device, dtype=torch.float32)
        self._tcp_offset_quaternion = torch.tensor(
            self.task_cfg.tcp_offset_rot, device=self.device, dtype=torch.float32
        )
        self._joint_seed = (
            wp.to_torch(self._ik_model.joint_q)
            .to(device=self.device, dtype=torch.float32)
            .repeat(self.cfg.batch_size, 1)
        )
        self._joint_seed[:, self._arm_coordinate_ids] = self._arm_home

        target_name = "reset_dataset_tcp"
        objectives = [
            NewtonIKPoseObjectiveCfg(
                body_name=self.task_cfg.tcp_body_name,
                name=target_name,
                body_offset_pos=self.task_cfg.tcp_offset_pos,
                body_offset_rot=self.task_cfg.tcp_offset_rot,
                position_weight=100.0,
                rotation_weight=5.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=1.0),
        ]
        self._ik_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=self.cfg.ik_seeds,
                noise_std=self.cfg.ik_noise_std,
                iterations=self.cfg.ik_iterations,
                lambda_initial=0.1,
                rng_seed=self.cfg.seed,
            ),
            model=self._ik_model,
            num_envs=self.cfg.batch_size,
            device=str(self.device),
            objectives=objectives,
            link_resolver=lambda _name: self._hand_id,
        )
        self._pose_objective = self._ik_solver.objectives_by_name[target_name]

        source_half = torch.tensor(self.task_cfg.cup_grasp_box_half, device=self.device, dtype=torch.float32)
        self._source_half = source_half
        self._cup_center_offset = torch.tensor((0.0, 0.0, float(source_half[2])), device=self.device)
        self._cup_grasp_offset = torch.tensor(
            (0.0, 0.0, float(self.task_cfg.cup_grasp_height)),
            device=self.device,
        )
        source_inner_lower = torch.as_tensor(self.env._source_inner_lo, device=self.device, dtype=torch.float32)
        source_inner_upper = torch.as_tensor(self.env._source_inner_hi, device=self.device, dtype=torch.float32)
        local_particles = self.env._media_local_points_t.to(self.device)
        if not bool(((local_particles >= source_inner_lower) & (local_particles <= source_inner_upper)).all()):
            raise RuntimeError("The existing particle sampler produced a point outside the source-cup cavity.")
        nominal_source = torch.tensor(self.task_cfg.cup_reset_pos, device=self.device, dtype=torch.float32)
        self._source_region_center = nominal_source + self._cup_center_offset
        self._target_half = torch.tensor(
            (
                self.task_cfg.target_cup_inner_width * 0.5 + self.task_cfg.target_cup_wall_thickness,
                self.task_cfg.target_cup_inner_depth * 0.5 + self.task_cfg.target_cup_wall_thickness,
                (self.task_cfg.target_cup_cavity_depth + self.task_cfg.target_cup_bottom_thickness) * 0.5,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self._target_rim_height = float(self._target_half[2] * 2.0)

    def _sample_category(
        self,
        category: int,
        required_count: int,
        *,
        near_pour: bool = False,
    ) -> dict[str, torch.Tensor]:
        if required_count <= 0:
            raise ValueError("A sampled reset category must request at least one state.")
        if near_pour and category != GRASPING_CATEGORY:
            raise ValueError("Only grasping states can use the near-pour proposal region.")
        accepted_count = 0
        accepted_side_counts = torch.zeros(4, device=self.device, dtype=torch.long)
        side_quotas = torch.full((4,), required_count // 4, device=self.device, dtype=torch.long)
        side_quotas[: required_count % 4] += 1
        max_attempts = required_count * self.cfg.max_attempt_multiplier
        initial_attempt_count = self._attempt_counts[category]

        def evaluate_batch(candidate_ids: range) -> dict[str, torch.Tensor]:
            nonlocal accepted_count
            candidate_count = len(candidate_ids)
            proposal = self._propose_batch(
                category,
                accepted_side_counts,
                side_quotas,
                near_pour=near_pour,
                count=candidate_count,
            )
            self._attempt_counts[category] += candidate_count
            valid = self._validate_proposal(proposal, category)
            valid_before_quota = valid.clone()
            if category == GRASPING_CATEGORY:
                keep = torch.zeros_like(valid)
                for side in range(4):
                    side_rows = torch.where(valid & (proposal["grasp_side"] == side))[0]
                    remaining = int(side_quotas[side] - accepted_side_counts[side])
                    chosen = side_rows[: max(remaining, 0)]
                    keep[chosen] = True
                    accepted_side_counts[side] += chosen.numel()
                valid = keep
            else:
                valid_rows = torch.where(valid)[0][: required_count - accepted_count]
                valid = torch.zeros_like(valid)
                valid[valid_rows] = True
            self._rejection_counts[category]["quota_full"] += int((valid_before_quota & ~valid).sum())
            accepted = int(valid.sum())
            accepted_count += accepted
            return {key: proposal[key][valid].detach().clone() for key in _STATE_KEYS}

        def batch_count(batch: dict[str, torch.Tensor]) -> int:
            return int(batch["category"].shape[0])

        def batch_slice(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
            return {key: value[:count] for key, value in batch.items()}

        try:
            batches, _ = reset_dataset_collect_batches(
                required_count,
                batch_size=self.cfg.batch_size,
                max_candidate_count=max_attempts,
                evaluate_batch=evaluate_batch,
                batch_count=batch_count,
                batch_slice=batch_slice,
            )
        except RuntimeError as error:
            attempted = self._attempt_counts[category] - initial_attempt_count
            raise RuntimeError(
                f"Reset-dataset sampling exhausted {attempted} {self._category_name(category)} candidates "
                f"after accepting {accepted_count}/{required_count}; rejection counts are "
                f"{dict(self._rejection_counts[category])}."
            ) from error
        return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in _STATE_KEYS}

    def _propose_batch(
        self,
        category: int,
        accepted_side_counts: torch.Tensor,
        side_quotas: torch.Tensor,
        *,
        near_pour: bool = False,
        count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        count = self.cfg.batch_size if count is None else count
        attempt_ids = torch.arange(
            self._next_attempt_id, self._next_attempt_id + count, device=self.device, dtype=torch.int64
        )
        self._next_attempt_id += count
        target_positions = self._sample_target_positions(count)
        target_quaternions = self._identity_quaternions(count)
        finger_position = torch.empty((count, 2), device=self.device)

        if category == GRASPING_CATEGORY:
            side_deficit = side_quotas - accepted_side_counts
            active_sides = torch.where(side_deficit > 0)[0]
            side_order = active_sides[torch.argsort(side_deficit[active_sides], descending=True)]
            grasp_side = side_order[torch.arange(count, device=self.device) % side_order.numel()].to(torch.int8)
            side_angle = grasp_side.to(torch.float32) * (0.5 * math.pi)
            side_axis = torch.zeros((count, 3), device=self.device)
            side_axis[:, 2] = 1.0
            side_rotation = math_utils.quat_from_angle_axis(side_angle, side_axis)
            base_grasp = torch.tensor(
                self.task_cfg.cup_grasp_tcp_quat_c, device=self.device, dtype=torch.float32
            ).expand(count, -1)
            cup_to_tcp = math_utils.quat_mul(side_rotation, base_grasp)
            offset_tcp = torch.randn((count, 3), device=self.device, generator=self.generator)
            offset_tcp *= torch.tensor(self.cfg.grasp_position_std, device=self.device)
            if near_pour:
                source_quaternions, cup_centers = self._sample_near_pour_cup_poses(target_positions)
                tcp_quaternions = math_utils.quat_unique(math_utils.quat_mul(source_quaternions, cup_to_tcp))
                source_positions = cup_centers - math_utils.quat_apply(
                    source_quaternions,
                    self._cup_center_offset.expand(count, -1),
                )
                grasp_positions = source_positions + math_utils.quat_apply(
                    source_quaternions,
                    self._cup_grasp_offset.expand(count, -1),
                )
                tcp_positions = grasp_positions - math_utils.quat_apply(tcp_quaternions, offset_tcp)
            else:
                tcp_positions = self._sample_workspace_positions(count)
                tcp_quaternions = self._sample_uniform_quaternions(count)
                source_quaternions = math_utils.quat_unique(
                    math_utils.quat_mul(tcp_quaternions, math_utils.quat_conjugate(cup_to_tcp))
                )
                source_positions = source_root_position_from_tcp_grasp(
                    tcp_positions,
                    tcp_quaternions,
                    source_quaternions,
                    self._cup_grasp_offset,
                    offset_tcp,
                )
            finger_position.fill_(float(self._source_half[1]))
            # Restore the physical fingers tangent to the cup, but command the task's full close
            # target from the first physics step. Initializing the physical joints at that target
            # would embed the collision meshes in the cup; separating q from its target creates a
            # strong bilateral preload without reset penetration.
            finger_target = torch.full_like(
                finger_position,
                float(self.task_cfg.actions.gripper_action.close_position),
            )
            grasp_region = torch.full((count,), int(near_pour), device=self.device, dtype=torch.int8)
        else:
            tcp_positions = self._sample_workspace_positions(count)
            tcp_quaternions = self._sample_uniform_quaternions(count)
            grasp_side = torch.full((count,), -1, device=self.device, dtype=torch.int8)
            grasp_region = torch.full((count,), -1, device=self.device, dtype=torch.int8)
            source_positions, source_quaternions = self._sample_table_source_poses(count)
            finger_position.uniform_(*self._finger_position_range, generator=self.generator)
            # The two fingers are driven symmetrically in the task; one sampled physical opening
            # therefore maps to equal joint coordinates rather than two unrelated finger widths.
            finger_position[:, 1] = finger_position[:, 0]
            finger_target = finger_position.clone()

        (
            robot_q,
            ik_cost,
            position_residual,
            rotation_residual,
            actual_tcp_position,
            actual_tcp_quaternion,
            ik_solution_valid,
        ) = self._solve_ik(tcp_positions, tcp_quaternions, finger_position)
        source_pose = torch.cat((source_positions, source_quaternions), dim=-1)
        target_pose = torch.cat((target_positions, target_quaternions), dim=-1)
        objective_components = torch.full((count, 3), -1.0, device=self.device)
        objective_raw = torch.full((count,), -1.0, device=self.device)
        objective = torch.full((count,), -1.0, device=self.device)
        if category == GRASPING_CATEGORY:
            objective_components = grasp_objective_components(
                source_pose,
                target_pose,
                source_region_center=self._source_region_center,
                cup_center_offset=self._cup_center_offset,
                target_rim_height=self._target_rim_height,
                distance_threshold=self.cfg.objective_distance_threshold,
                target_horizontal_threshold=self.cfg.objective_target_horizontal_threshold,
                target_height_threshold=self.cfg.objective_target_height_threshold,
                inversion_gate_horizontal_threshold=self.cfg.objective_inversion_gate_horizontal_threshold,
            )
            weights = torch.tensor(self.cfg.objective_weights, device=self.device)
            objective_raw = (objective_components * weights).sum(dim=-1)
            objective = objective_raw.clone()

        return {
            "arm_joint_position": robot_q[:, self._arm_coordinate_ids],
            "arm_joint_velocity": torch.zeros((count, 7), device=self.device),
            "finger_joint_position": finger_position,
            "finger_joint_velocity": torch.zeros((count, 2), device=self.device),
            "finger_joint_target": finger_target,
            "source_root_pose": source_pose,
            "source_root_velocity": torch.zeros((count, 6), device=self.device),
            "target_root_pose": target_pose,
            "target_root_velocity": torch.zeros((count, 6), device=self.device),
            "category": torch.full((count,), category, device=self.device, dtype=torch.int8),
            "objective": objective,
            "objective_raw": objective_raw,
            "objective_components": objective_components,
            "grasp_region": grasp_region,
            "grasp_side": grasp_side,
            "attempt_id": attempt_ids,
            "particle_layout_id": torch.zeros(count, device=self.device, dtype=torch.int32),
            "ik_cost": ik_cost,
            "ik_position_residual": position_residual,
            "ik_rotation_residual": rotation_residual,
            "_robot_q": robot_q,
            "_tcp_target_position": tcp_positions,
            "_tcp_target_quaternion": tcp_quaternions,
            "_tcp_actual_position": actual_tcp_position,
            "_tcp_actual_quaternion": actual_tcp_quaternion,
            "_ik_solution_valid": ik_solution_valid,
        }

    def _solve_ik(
        self, tcp_positions: torch.Tensor, tcp_quaternions: torch.Tensor, finger_position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import warp as wp

        count = int(tcp_positions.shape[0])
        if not 0 < count <= self.cfg.batch_size:
            raise ValueError(f"IK batch count must be in [1, {self.cfg.batch_size}], got {count}.")
        if tcp_quaternions.shape != (count, 4) or finger_position.shape != (count, 2):
            raise ValueError("IK targets and finger positions have inconsistent batch dimensions.")

        # NewtonIK is constructed once at the configured maximum batch size. Pad only the final
        # rejection-sampling batch and discard its duplicate rows after solving; rebuilding the
        # solver for a short tail batch would be both expensive and unnecessary.
        if count < self.cfg.batch_size:
            padding = self.cfg.batch_size - count
            tcp_positions = torch.cat((tcp_positions, tcp_positions[-1:].expand(padding, -1)), dim=0)
            tcp_quaternions = torch.cat((tcp_quaternions, tcp_quaternions[-1:].expand(padding, -1)), dim=0)
            finger_position_padded = torch.cat((finger_position, finger_position[-1:].expand(padding, -1)), dim=0)
        else:
            finger_position_padded = finger_position

        self._pose_objective.position_objective.set_target_positions(
            wp.from_torch(tcp_positions.contiguous(), dtype=wp.vec3)
        )
        self._pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(tcp_quaternions.contiguous(), dtype=wp.vec4)
        )
        seed = self._joint_seed.clone()
        seed[:, self._finger_coordinate_ids] = finger_position_padded
        self._ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))
        expanded_joint_q = wp.to_torch(self._ik_solver.joint_q).reshape(self.cfg.batch_size, self.cfg.ik_seeds, -1)
        expanded_cost = wp.to_torch(self._ik_solver.costs).reshape(self.cfg.batch_size, self.cfg.ik_seeds)
        residuals = wp.to_torch(self._ik_solver.solver.residuals).reshape(self.cfg.batch_size, self.cfg.ik_seeds, -1)
        expanded_position_residual = torch.linalg.vector_norm(residuals[:, :, :3] / 100.0, dim=-1)
        expanded_rotation_residual = torch.linalg.vector_norm(residuals[:, :, 3:6] / 5.0, dim=-1)
        expanded_arm_q = expanded_joint_q[:, :, self._arm_coordinate_ids]
        arm_margin = torch.minimum(
            expanded_arm_q - self._arm_limits[:, 0], self._arm_limits[:, 1] - expanded_arm_q
        ).amin(dim=-1)
        seed_valid = (
            torch.isfinite(expanded_joint_q).all(dim=-1)
            & torch.isfinite(expanded_cost)
            & torch.isfinite(expanded_position_residual)
            & torch.isfinite(expanded_rotation_residual)
            & (expanded_cost <= self.cfg.ik_max_cost)
            & (arm_margin >= self.cfg.ik_joint_margin)
            & (expanded_position_residual <= self.cfg.ik_max_position_residual)
            & (expanded_rotation_residual <= self.cfg.ik_max_rotation_residual)
            & (torch.linalg.vector_norm(expanded_arm_q - self._arm_home, dim=-1) <= self.cfg.ik_max_home_distance)
        )
        valid_cost = expanded_cost.masked_fill(~seed_valid, torch.inf)
        best_seed = valid_cost.argmin(dim=-1)
        solution_valid = seed_valid.any(dim=-1)
        fallback_seed = torch.nan_to_num(expanded_cost, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(
            dim=-1
        )
        best_seed = torch.where(solution_valid, best_seed, fallback_seed)
        rows = torch.arange(count, device=self.device)
        selected_seed = best_seed[rows]
        solved = expanded_joint_q[rows, selected_seed].clone()
        solved[:, self._finger_coordinate_ids] = finger_position
        cost = expanded_cost[rows, selected_seed].clone()
        position_residual = expanded_position_residual[rows, selected_seed].clone()
        rotation_residual = expanded_rotation_residual[rows, selected_seed].clone()
        solution_valid = solution_valid[rows]
        body_poses = wp.to_torch(self._ik_solver.solver.body_q).reshape(self.cfg.batch_size, self.cfg.ik_seeds, -1, 7)
        hand_pose = body_poses[rows, selected_seed, self._hand_id].clone()
        actual_tcp_position, actual_tcp_quaternion = math_utils.combine_frame_transforms(
            hand_pose[:, :3],
            hand_pose[:, 3:7],
            self._tcp_offset_pos.expand(count, -1),
            self._tcp_offset_quaternion.expand(count, -1),
        )
        return (
            solved,
            cost,
            position_residual,
            rotation_residual,
            actual_tcp_position,
            actual_tcp_quaternion,
            solution_valid,
        )

    def _validate_proposal(self, proposal: dict[str, torch.Tensor], category: int) -> torch.Tensor:
        count = int(proposal["category"].shape[0])
        valid = torch.ones(count, device=self.device, dtype=torch.bool)
        arm_q = proposal["arm_joint_position"]
        arm_margin = torch.minimum(arm_q - self._arm_limits[:, 0], self._arm_limits[:, 1] - arm_q).amin(dim=-1)
        checks = (
            ("ik_no_valid_seed", proposal["_ik_solution_valid"]),
            ("ik_nonfinite", torch.isfinite(proposal["_robot_q"]).all(dim=-1) & torch.isfinite(proposal["ik_cost"])),
            ("ik_cost", proposal["ik_cost"] <= self.cfg.ik_max_cost),
            ("ik_joint_limit", arm_margin >= self.cfg.ik_joint_margin),
            ("ik_position_residual", proposal["ik_position_residual"] <= self.cfg.ik_max_position_residual),
            ("ik_rotation_residual", proposal["ik_rotation_residual"] <= self.cfg.ik_max_rotation_residual),
            (
                "ik_discontinuity",
                torch.linalg.vector_norm(arm_q - self._arm_home, dim=-1) <= self.cfg.ik_max_home_distance,
            ),
        )
        for reason, check in checks:
            valid = self._reject(valid, check, category, reason)

        source_pose = proposal["source_root_pose"]
        target_pose = proposal["target_root_pose"]
        source_center = source_pose[:, :3] + math_utils.quat_apply(
            source_pose[:, 3:7], self._cup_center_offset.expand(count, -1)
        )
        target_center = target_pose[:, :3].clone()
        target_center[:, 2] += self._target_half[2]
        source_target_clear = ~oriented_boxes_overlap(
            source_center,
            source_pose[:, 3:7],
            self._source_half,
            target_center,
            target_pose[:, 3:7],
            self._target_half,
            clearance=self.cfg.obstacle_clearance,
        )
        valid = self._reject(valid, source_target_clear, category, "source_target_collision")
        valid = self._reject(valid, self._box_above_table(source_pose), category, "source_table_collision")
        valid = self._reject(valid, self._box_above_table(target_pose, target=True), category, "target_table_collision")
        valid = self._reject(
            valid,
            oriented_box_supported_by_bounds(
                target_pose,
                self._target_half,
                self._tabletop_support_lower_xy,
                self._tabletop_support_upper_xy,
                clearance=self.cfg.obstacle_clearance,
            ),
            category,
            "target_table_support",
        )
        if category == NON_GRASPING_CATEGORY:
            valid = self._reject(
                valid,
                oriented_box_supported_by_bounds(
                    source_pose,
                    self._source_half,
                    self._tabletop_support_lower_xy,
                    self._tabletop_support_upper_xy,
                    clearance=self.cfg.obstacle_clearance,
                ),
                category,
                "source_table_support",
            )
        valid = self._reject(valid, self._particles_in_workspace(source_pose), category, "particle_workspace")
        if category == GRASPING_CATEGORY and bool((proposal["grasp_region"] == 1).any()):
            valid = self._reject(
                valid,
                above_target_tilted_mask(
                    source_pose,
                    target_pose,
                    cup_center_offset=self._cup_center_offset,
                    target_rim_height=self._target_rim_height,
                    max_horizontal_distance=self.cfg.near_pour_horizontal_radius,
                    min_vertical_clearance=self.cfg.near_pour_height_range[0],
                    min_tilt_angle=self.cfg.near_pour_tilt_angle_range[0],
                ),
                category,
                "invalid_near_pour_geometry",
            )
        if category == GRASPING_CATEGORY:
            source_grasp_point = source_pose[:, :3] + math_utils.quat_apply(
                source_pose[:, 3:7],
                self._cup_grasp_offset.expand(count, -1),
            )
            cup_offset_tcp = math_utils.quat_apply_inverse(
                proposal["_tcp_actual_quaternion"],
                source_grasp_point - proposal["_tcp_actual_position"],
            )
            seating_bound = torch.tensor(self.cfg.grasp_seating_max_offset, device=self.device)
            side_angle = proposal["grasp_side"].to(torch.float32) * (0.5 * math.pi)
            side_axis = torch.zeros((count, 3), device=self.device)
            side_axis[:, 2] = 1.0
            expected_cup_to_tcp = math_utils.quat_mul(
                math_utils.quat_from_angle_axis(side_angle, side_axis),
                torch.tensor(self.task_cfg.cup_grasp_tcp_quat_c, device=self.device).expand(count, -1),
            )
            actual_cup_to_tcp = math_utils.quat_mul(
                math_utils.quat_conjugate(source_pose[:, 3:7]), proposal["_tcp_actual_quaternion"]
            )
            seating_rotation_error = math_utils.quat_error_magnitude(actual_cup_to_tcp, expected_cup_to_tcp)
            valid = self._reject(
                valid,
                (cup_offset_tcp.abs() <= seating_bound).all(dim=-1)
                & (seating_rotation_error <= self.cfg.grasp_seating_max_rotation_error),
                category,
                "invalid_grasp_seating",
            )
        else:
            cup_tcp_distance = torch.linalg.vector_norm(source_center - proposal["_tcp_actual_position"], dim=-1)
            valid = self._reject(
                valid,
                cup_tcp_distance >= self.cfg.non_grasping_min_tcp_source_distance,
                category,
                "source_inside_gripper",
            )

        # A grasp candidate gets the required robot-only screen before inserting the source cup.
        # The source proxy is parked far outside the workspace for this first pass.
        if category == GRASPING_CATEGORY:
            pre_indices = torch.where(valid)[0]
            if pre_indices.numel():
                far_source = proposal["source_root_pose"][pre_indices, :3].clone()
                far_source[:, 2] = 10.0
                pre_clear = self._collision_screen(
                    proposal["_robot_q"][pre_indices],
                    far_source,
                    proposal["source_root_pose"][pre_indices, 3:7],
                    proposal["target_root_pose"][pre_indices, :3],
                    allow_finger_contact=False,
                )
                full_check = torch.zeros_like(valid)
                full_check[pre_indices] = pre_clear
                valid = self._reject(valid, full_check, category, "robot_preinsert_collision")

        final_indices = torch.where(valid)[0]
        if final_indices.numel():
            collision_clear = self._collision_screen(
                proposal["_robot_q"][final_indices],
                proposal["source_root_pose"][final_indices, :3],
                proposal["source_root_pose"][final_indices, 3:7],
                proposal["target_root_pose"][final_indices, :3],
                allow_finger_contact=category == GRASPING_CATEGORY,
            )
            full_check = torch.zeros_like(valid)
            full_check[final_indices] = collision_clear
            valid = self._reject(valid, full_check, category, "complete_state_collision")
        return valid

    def _collision_screen(
        self,
        robot_q: torch.Tensor,
        source_position: torch.Tensor,
        source_quaternion: torch.Tensor,
        target_position: torch.Tensor,
        *,
        allow_finger_contact: bool,
    ) -> torch.Tensor:
        from ._reset_collision_screen import collision_free_reset_candidates

        return collision_free_reset_candidates(
            self._prototype_builder,
            robot_q,
            source_position,
            source_quaternion,
            target_position,
            source_box_half=tuple(float(value) for value in self._source_half),
            target_vertices=self.env._target_vertices,
            target_indices=self.env._target_indices,
            collider_margin=float(self.task_cfg.collider_margin),
            device=str(self.device),
            penetration_tolerance=self.cfg.collision_penetration_tolerance,
            allow_source_finger_contact=allow_finger_contact,
            source_finger_penetration_tolerance=self.cfg.finger_contact_penetration_tolerance,
            check_self_collision=True,
            check_complete_robot_table=True,
        )

    def _box_above_table(self, pose: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        half = self._target_half if target else self._source_half
        center = pose[:, :3] + math_utils.quat_apply(
            pose[:, 3:7], torch.tensor((0.0, 0.0, float(half[2])), device=self.device).expand(pose.shape[0], -1)
        )
        rotation = math_utils.matrix_from_quat(pose[:, 3:7]).abs()
        vertical_radius = (rotation[:, 2, :] * half).sum(dim=-1)
        return center[:, 2] - vertical_radius >= -self.cfg.collision_penetration_tolerance

    def _particles_in_workspace(self, source_pose: torch.Tensor) -> torch.Tensor:
        # Reuse the task's existing particle-sampling transform rather than maintaining a second
        # interpretation of the cup-local media layout in this offline tool.
        world = self.env._sample_cup_media(source_pose[:, :3], source_pose[:, 3:7])
        lower = torch.tensor(self.task_cfg.particle_workspace_lower_bound, device=self.device)
        upper = torch.tensor(self.task_cfg.particle_workspace_upper_bound, device=self.device)
        return ((world >= lower) & (world <= upper)).all(dim=-1).all(dim=-1)

    def _sample_workspace_positions(self, count: int) -> torch.Tensor:
        fraction = self.cfg.workspace_central_fraction

        def central(bounds: tuple[float, float]) -> tuple[float, float]:
            midpoint = 0.5 * (bounds[0] + bounds[1])
            half = 0.5 * (bounds[1] - bounds[0]) * fraction
            return midpoint - half, midpoint + half

        radius_lower, radius_upper = central(self.cfg.workspace_radius_range)
        # Azimuth is periodic rather than a kinematic limit.  Preserve the complete configured
        # circle and apply the central-fraction margin only to radial reach and height.
        angle_lower, angle_upper = self.cfg.workspace_azimuth_range
        height_lower, height_upper = central(self.cfg.workspace_height_range)
        radius_square = torch.empty(count, device=self.device).uniform_(
            radius_lower**2, radius_upper**2, generator=self.generator
        )
        radius = torch.sqrt(radius_square)
        angle = torch.empty(count, device=self.device).uniform_(angle_lower, angle_upper, generator=self.generator)
        height = torch.empty(count, device=self.device).uniform_(height_lower, height_upper, generator=self.generator)
        return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), height), dim=-1)

    def _sample_uniform_quaternions(self, count: int) -> torch.Tensor:
        # Shoemake's transform produces Haar-uniform SO(3) rotations in XYZW order.
        random = torch.rand((count, 3), device=self.device, generator=self.generator)
        first = torch.sqrt(1.0 - random[:, 0])
        second = torch.sqrt(random[:, 0])
        quaternion = torch.stack(
            (
                first * torch.sin(2.0 * math.pi * random[:, 1]),
                first * torch.cos(2.0 * math.pi * random[:, 1]),
                second * torch.sin(2.0 * math.pi * random[:, 2]),
                second * torch.cos(2.0 * math.pi * random[:, 2]),
            ),
            dim=-1,
        )
        return math_utils.quat_unique(quaternion)

    def _sample_table_source_poses(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        radius_range = self.task_cfg.curriculum_randomized_source_radius_range
        if radius_range is None:
            center = torch.tensor(self.task_cfg.cup_reset_pos, device=self.device)
            extent = torch.tensor(self.task_cfg.curriculum_randomized_source_position_range, device=self.device)
            random_offset = 2.0 * torch.rand((count, 2), device=self.device, generator=self.generator) - 1.0
            xy = center[:2] + random_offset * extent
        else:
            radius_square = torch.empty(count, device=self.device).uniform_(
                radius_range[0] ** 2, radius_range[1] ** 2, generator=self.generator
            )
            radius = torch.sqrt(radius_square)
            angle = torch.empty(count, device=self.device).uniform_(
                -self.task_cfg.curriculum_randomized_source_azimuth_range,
                self.task_cfg.curriculum_randomized_source_azimuth_range,
                generator=self.generator,
            )
            xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
        positions = torch.zeros((count, 3), device=self.device)
        positions[:, :2] = xy
        yaw = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.generator)
        quaternions = torch.zeros((count, 4), device=self.device)
        quaternions[:, 2] = torch.sin(0.5 * yaw)
        quaternions[:, 3] = torch.cos(0.5 * yaw)
        return positions, quaternions

    def _sample_target_positions(self, count: int) -> torch.Tensor:
        center = torch.tensor(self.task_cfg.curriculum_randomized_target_center_xy, device=self.device)
        extent = torch.tensor(self.task_cfg.curriculum_randomized_target_position_range, device=self.device)
        configured_lower = center - extent
        configured_upper = center + extent
        supported_lower = self._tabletop_support_lower_xy + self._target_half[:2] + self.cfg.obstacle_clearance
        supported_upper = self._tabletop_support_upper_xy - self._target_half[:2] - self.cfg.obstacle_clearance
        lower = torch.maximum(configured_lower, supported_lower)
        upper = torch.minimum(configured_upper, supported_upper)
        if not bool(torch.all(lower < upper)):
            raise RuntimeError("The configured target region has no fully supported tabletop area.")
        positions = torch.zeros((count, 3), device=self.device)
        positions[:, :2] = lower + torch.rand((count, 2), device=self.device, generator=self.generator) * (
            upper - lower
        )
        return positions

    def _sample_near_pour_cup_poses(self, target_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tilted cup centers directly above already-supported receiver poses."""
        count = target_positions.shape[0]
        azimuth = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.generator)
        tilt = torch.empty(count, device=self.device).uniform_(
            *self.cfg.near_pour_tilt_angle_range,
            generator=self.generator,
        )
        tilt_axes = torch.stack((torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)), dim=-1)
        source_quaternions = math_utils.quat_unique(math_utils.quat_from_angle_axis(tilt, tilt_axes))
        radial_unit = torch.rand(count, device=self.device, generator=self.generator)
        radial_distance = self.cfg.near_pour_horizontal_radius * torch.sqrt(radial_unit)
        radial_angle = torch.empty(count, device=self.device).uniform_(
            -math.pi,
            math.pi,
            generator=self.generator,
        )
        cup_centers = target_positions.clone()
        cup_centers[:, 0] += radial_distance * torch.cos(radial_angle)
        cup_centers[:, 1] += radial_distance * torch.sin(radial_angle)
        clearance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.near_pour_height_range,
            generator=self.generator,
        )
        cup_centers[:, 2] += self._target_rim_height + clearance
        return source_quaternions, cup_centers

    def _identity_quaternions(self, count: int) -> torch.Tensor:
        quaternions = torch.zeros((count, 4), device=self.device)
        quaternions[:, 3] = 1.0
        return quaternions

    def _reject(self, current_valid: torch.Tensor, check: torch.Tensor, category: int, reason: str) -> torch.Tensor:
        rejected = current_valid & ~check
        self._rejection_counts[category][reason] += int(rejected.sum())
        return current_valid & check

    def _task_contract(self) -> dict[str, Any]:
        """Return the same canonical contract checked by the runtime loader."""
        return build_franka_pour_reset_task_contract(self.env)

    @staticmethod
    def _category_name(category: int, *, near_pour: bool = False) -> str:
        if category == GRASPING_CATEGORY:
            return "near-pour grasping" if near_pour else "broad grasping"
        return "non-grasping"
