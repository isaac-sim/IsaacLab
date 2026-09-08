# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the Franka Pour reset artifact with Newton IK and collision rejection.

The generator intentionally lives in one file. It creates the same 14-phase, 20,000-row
distribution used by the reference artifact, while keeping only the validation needed to make
individual resets safe: IK residuals, joint limits, task geometry, and Newton collision queries.
The artifact stores no simulated particle trajectory; it stores one cup-local particle layout and
rigid reset states that the environment expands when an episode starts.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import newton
import torch
import warp as wp
from isaaclab_newton.cloner import copy_newton_clone_source

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_pour.geometry import (
    CUP_GRASP_HEIGHT,
    CUP_GRASP_TCP_QUAT_C,
    SOURCE_CUP_GEOMETRY,
    TARGET_CUP_GEOMETRY,
    box_above_plane,
    box_center_from_base_pose,
    box_supported,
    oriented_boxes_overlap,
)
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FrankaPourResetDatasetEnvCfg,
    _reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_pour.reset_dataset_io import (
    FRANKA_POUR_RESET_DATASET_FORMAT,
    FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
    RESET_DATASET_STATE_NAMES,
    reset_dataset_content_digest,
    reset_dataset_validate_runtime,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "datasets/franka_pour/reset_dataset.pt"
_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

_NON_GRASPING = 0
_GRASPING = 1
_REACHING = 0
_NEAR_OBJECT = 1
_TRANSPORT = 2
_NEAR_GOAL = 3

(
    _REACHING_ALIGNED,
    _REACHING_BROAD,
    _CONTACT,
    _CAPTURE,
    _PREGRASP,
    _APPROACH,
    _JUST_GRASPED,
    _LOW_LIFT,
    _UPRIGHT,
    _PRE_POUR,
    _POUR_ENTRY,
    _BOUNDARY,
    _QUICK,
    _IMMEDIATE,
) = range(14)


@dataclass(frozen=True)
class _PhaseSpec:
    """One phase in the ordered reset manifold."""

    name: str
    region: int
    category: int
    weight: float
    path_index: int


_PHASES = (
    _PhaseSpec("reaching_aligned", _REACHING, _NON_GRASPING, 1.0 / 6.0, 1),
    _PhaseSpec("reaching_broad", _REACHING, _NON_GRASPING, 1.0 / 6.0, 0),
    _PhaseSpec("near_object_contact", _NEAR_OBJECT, _NON_GRASPING, 1.0 / 6.0, 4),
    _PhaseSpec("near_object_capture", _NEAR_OBJECT, _NON_GRASPING, 1.0 / 6.0, 5),
    _PhaseSpec("near_object_pregrasp", _NEAR_OBJECT, _NON_GRASPING, 1.0 / 6.0, 3),
    _PhaseSpec("near_object_approach", _NEAR_OBJECT, _NON_GRASPING, 1.0 / 6.0, 2),
    _PhaseSpec("transport_just_grasped", _TRANSPORT, _GRASPING, 0.14, 6),
    _PhaseSpec("transport_low_lift", _TRANSPORT, _GRASPING, 0.14, 7),
    _PhaseSpec("transport_upright", _TRANSPORT, _GRASPING, 0.14, 8),
    _PhaseSpec("transport_pre_pour", _TRANSPORT, _GRASPING, 0.14, 9),
    _PhaseSpec("transport_pour_entry", _TRANSPORT, _GRASPING, 0.14, 10),
    _PhaseSpec("near_goal_boundary", _NEAR_GOAL, _GRASPING, 0.26, 11),
    _PhaseSpec("near_goal_quick", _NEAR_GOAL, _GRASPING, 0.03, 12),
    _PhaseSpec("near_goal_immediate", _NEAR_GOAL, _GRASPING, 0.01, 13),
)
_PHASE_NAMES = tuple(phase.name for phase in _PHASES)
_PHASE_ORDER = tuple(sorted(range(len(_PHASES)), key=lambda phase: _PHASES[phase].path_index))
_TCP_SAMPLING = {
    _REACHING_ALIGNED: ("reaching_aligned_radius_range", None, None),
    _REACHING_BROAD: ("reaching_broad_radius_range", "reaching_broad_cone_angle", "reaching_broad_rotation_error"),
    _CONTACT: ("contact_radius_range", None, None),
    _CAPTURE: ("capture_radius_range", "capture_cone_angle", "capture_rotation_error"),
    _PREGRASP: ("pregrasp_radius_range", None, None),
    _APPROACH: ("approach_radius_range", None, None),
}
_FINGER_MODES = {_CONTACT: "contact", _CAPTURE: "capture"}
_TRANSPORT_SAMPLING = {
    _JUST_GRASPED: ("pickup", None, (0.0, 0.02), (0.0, math.radians(5.0)), 0.005),
    _LOW_LIFT: ("pickup", None, (0.01, 0.08), (0.0, math.radians(10.0)), 0.01),
    _UPRIGHT: ("interpolate", None, (0.06, 0.18), (0.0, math.radians(20.0)), 0.015),
    _PRE_POUR: ("goal", (0.03, 0.10), (0.12, 0.25), (math.radians(20.0), math.radians(90.0)), 0.0),
    _POUR_ENTRY: ("goal", (0.015, 0.06), (0.12, 0.22), (math.radians(90.0), math.radians(130.0)), 0.0),
}
_NEAR_GOAL_SAMPLING = {
    _BOUNDARY: ("boundary", None, None, False),
    _QUICK: ("goal", (0.0, 0.5), (0.0, 0.6), True),
    _IMMEDIATE: ("goal", (0.0, 0.015), (0.2, 1.0), False),
}


@dataclass(frozen=True)
class GeneratorCfg:
    """Controls the reference reset distribution and bounded rejection loop."""

    grasping_count: int = 10_000
    non_grasping_count: int = 10_000
    batch_size: int = 256
    seed: int = 42
    max_attempt_multiplier: int = 100
    grasp_side_ids: tuple[int, ...] = (0, 1, 2, 3)

    source_radius_range: tuple[float, float] = (0.40, 0.78)
    source_azimuth_range: float = math.radians(35.0)
    source_target_distance_range: tuple[float, float] = (0.15, 0.35)
    pour_source_offset_xy: tuple[float, float] = (0.0, 0.05)
    reaching_aligned_radius_range: tuple[float, float] = (0.04, 0.30)
    reaching_broad_radius_range: tuple[float, float] = (0.12, 0.32)
    contact_radius_range: tuple[float, float] = (0.001, 0.01)
    capture_radius_range: tuple[float, float] = (0.002, 0.015)
    pregrasp_radius_range: tuple[float, float] = (0.005, 0.025)
    approach_radius_range: tuple[float, float] = (0.02, 0.08)
    reaching_broad_cone_angle: float = math.radians(65.0)
    reaching_broad_rotation_error: float = math.radians(45.0)
    capture_cone_angle: float = math.radians(15.0)
    capture_rotation_error: float = math.radians(5.0)
    capture_filter_step_range: tuple[int, int] = (2, 18)
    contact_finger_clearance: float = 0.002

    near_pour_horizontal_radius: float = 0.04
    near_pour_height_range: tuple[float, float] = (0.18, 0.22)
    near_pour_tilt_range: tuple[float, float] = (math.radians(145.0), math.radians(170.0))
    boundary_tilt_range: tuple[float, float] = (math.radians(130.0), math.radians(138.0))

    ik_seeds: int = 64
    ik_iterations: int = 160
    ik_noise_std: float = 0.75
    ik_max_cost: float = 1.0e-3
    ik_joint_margin: float = 0.015
    ik_max_position_residual: float = 0.003
    ik_max_rotation_residual: float = math.radians(3.0)
    ik_max_home_distance: float = 6.0

    grasp_position_std: tuple[float, float, float] = (0.0015, 0.00005, 0.0015)
    near_goal_grasp_position_std: tuple[float, float, float] = (0.0005, 0.00002, 0.0005)
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

    def __post_init__(self) -> None:
        for name in ("grasping_count", "non_grasping_count", "batch_size", "max_attempt_multiplier"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.non_grasping_count < 6 or self.grasping_count < 100:
            raise ValueError("At least 6 non-grasping and 100 grasping rows are needed to cover all 14 phases.")
        if not self.grasp_side_ids or len(set(self.grasp_side_ids)) != len(self.grasp_side_ids):
            raise ValueError("grasp_side_ids must contain unique side IDs.")
        if any(
            not isinstance(side, int) or isinstance(side, bool) or not 0 <= side <= 3 for side in self.grasp_side_ids
        ):
            raise ValueError("grasp_side_ids must contain integers in [0, 3].")


def _weighted_counts(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    raw = [total * weight for weight in weights]
    counts = [math.floor(value) for value in raw]
    order = sorted(range(len(weights)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return tuple(counts)


def phase_counts(cfg: GeneratorCfg) -> tuple[int, ...]:
    """Return the exact phase quotas used by the reference artifact."""
    counts = [0] * len(_PHASE_NAMES)
    for category, total in ((_NON_GRASPING, cfg.non_grasping_count), (_GRASPING, cfg.grasping_count)):
        phases = tuple(phase for phase in _PHASE_ORDER if _PHASES[phase].category == category)
        for phase, count in zip(
            phases,
            _weighted_counts(total, tuple(_PHASES[phase].weight for phase in phases)),
            strict=True,
        ):
            counts[phase] = count
    return tuple(counts)


def _normalize_progress(raw: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    progress = torch.empty_like(raw)
    for path_index, phase in enumerate(_PHASE_ORDER):
        rows = torch.where(phases == phase)[0]
        if rows.numel() == 0:
            continue
        ordered = rows[torch.argsort(raw[rows], descending=True, stable=True)]
        local = (
            raw.new_tensor((0.5,)) if len(ordered) == 1 else torch.linspace(0.0, 1.0, len(ordered), device=raw.device)
        )
        progress[ordered] = (path_index + local) / len(_PHASE_ORDER)
    progress[torch.argmin(progress)] = 0.0
    progress[torch.argmax(progress)] = 1.0
    return progress


def _grasp_objective(
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    source_region_center: torch.Tensor,
    cup_center_offset: torch.Tensor,
    target_rim_height: float,
    cfg: GeneratorCfg,
) -> torch.Tensor:
    count = source_pose.shape[0]
    cup_center = source_pose[:, :3] + math_utils.quat_apply(source_pose[:, 3:7], cup_center_offset.expand(count, -1))
    distance = torch.linalg.vector_norm(cup_center - source_region_center, dim=-1)
    distance_score = (distance / cfg.objective_distance_threshold).clamp(0.0, 1.0)
    local_up = torch.zeros((count, 3), device=source_pose.device)
    local_up[:, 2] = 1.0
    inversion = ((1.0 - math_utils.quat_apply(source_pose[:, 3:7], local_up)[:, 2]) * 0.5).clamp(0.0, 1.0)
    horizontal = torch.linalg.vector_norm(cup_center[:, :2] - target_pose[:, :2], dim=-1)
    horizontal_score = (1.0 - horizontal / cfg.objective_target_horizontal_threshold).clamp(0.0, 1.0)
    target_rim_z = target_pose[:, 2] + target_rim_height
    height_score = ((cup_center[:, 2] - target_rim_z) / cfg.objective_target_height_threshold).clamp(0.0, 1.0)
    target_score = horizontal_score * height_score
    inversion *= target_score * (horizontal <= cfg.objective_inversion_gate_horizontal_threshold)
    return (distance_score + inversion + target_score) / 3.0


def _tabletop_bounds(env: FrankaPourEnv, source_env_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    prim = sim_utils.get_current_stage().GetPrimAtPath(f"{source_env_path.rstrip('/')}/Table/Collisions/Cube")
    if not prim.IsValid():
        raise RuntimeError("Could not find the SeattleLab tabletop collision prim.")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lower = torch.tensor(tuple(bounds.GetMin()), device=env.device)[:2] - env.env_origins[0, :2]
    upper = torch.tensor(tuple(bounds.GetMax()), device=env.device)[:2] - env.env_origins[0, :2]
    if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all() & torch.all(lower < upper)):
        raise RuntimeError("Could not derive finite tabletop support bounds.")
    return lower, upper


class _GeneratorEnv(FrankaPourEnv):
    """One-world scene that skips the artifact loader and RL managers."""

    def load_managers(self) -> None:
        self._bind_scene_assets()
        self.command_manager = None
        self.reward_manager = None
        self.termination_manager = None
        self.curriculum_manager = None
        self.recorder_manager = None
        self.action_manager = None
        self.observation_manager = None


_TABLE_TEST_BODIES = {
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
}


@wp.kernel(enable_backward=False)
def _mark_colliding_worlds(
    contact_count: wp.array(dtype=wp.int32),
    contact_max: int,
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_body: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    shape_obstacle: wp.array(dtype=wp.int32),
    shape_table: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_robot: wp.array(dtype=wp.int32),
    body_tests_table: wp.array(dtype=wp.int32),
    body_allows_source_contact: wp.array(dtype=wp.int32),
    shape_source: wp.array(dtype=wp.int32),
    shape_margin: wp.array(dtype=wp.float32),
    penetration_tolerance: float,
    source_finger_penetration_tolerance: float,
    colliding_worlds: wp.array(dtype=wp.int32),
):
    contact = wp.tid()
    if contact >= contact_max or contact >= contact_count[0]:
        return

    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    obstacle0 = shape_obstacle[shape0] != 0
    obstacle1 = shape_obstacle[shape1] != 0
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    if not obstacle0 and not obstacle1:
        if body0 < 0 or body1 < 0 or body0 == body1:
            return
        if body_robot[body0] == 0 or body_robot[body1] == 0:
            return
        world0 = body_world[body0]
        if world0 < 0 or world0 != body_world[body1]:
            return
        point0 = wp.transform_point(body_q[body0], contact_point0[contact])
        point1 = wp.transform_point(body_q[body1], contact_point1[contact])
        separation = wp.dot(contact_normal[contact], point1 - point0)
        separation = separation - contact_margin0[contact] - contact_margin1[contact]
        if separation < -penetration_tolerance:
            wp.atomic_max(colliding_worlds, world0, 1)
        return
    if obstacle0 and obstacle1:
        return

    obstacle_shape = shape0
    robot_body = body1
    if not obstacle0:
        obstacle_shape = shape1
        robot_body = body0
    if robot_body < 0 or body_robot[robot_body] == 0:
        return
    if shape_table[obstacle_shape] != 0 and body_tests_table[robot_body] == 0:
        return
    world = body_world[robot_body]
    if world < 0 or shape_world[obstacle_shape] != world:
        return

    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    if body0 >= 0:
        transform0 = body_q[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
    point0 = wp.transform_point(transform0, contact_point0[contact])
    point1 = wp.transform_point(transform1, contact_point1[contact])
    separation = wp.dot(contact_normal[contact], point1 - point0)
    separation = separation - contact_margin0[contact] - contact_margin1[contact]
    allowed = penetration_tolerance
    if shape_source[obstacle_shape] != 0 and body_allows_source_contact[robot_body] != 0:
        separation = separation + shape_margin[shape0] + shape_margin[shape1]
        allowed = source_finger_penetration_tolerance
    if separation < -allowed:
        wp.atomic_max(colliding_worlds, world, 1)


class _CollisionScreen:
    """Batch exact Newton collision queries in isolated candidate worlds."""

    def __init__(
        self,
        prototype: newton.ModelBuilder,
        cfg: GeneratorCfg,
        collider_margin: float,
        device: str,
    ):
        self.capacity = cfg.batch_size
        self.coordinate_count = prototype.joint_coord_count
        self.device = device
        self.penetration_tolerance = cfg.collision_penetration_tolerance
        self.source_finger_penetration_tolerance = cfg.finger_contact_penetration_tolerance
        source_half = SOURCE_CUP_GEOMETRY.outer_half_extents
        self._source_half_height = source_half[2]
        target_boxes = TARGET_CUP_GEOMETRY.collision_boxes
        self._target_offsets = torch.tensor([box[0] for box in target_boxes], device=device)
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=float(collider_margin),
            has_shape_collision=True,
            has_particle_collision=False,
            is_visible=False,
        )
        parked = wp.transform(wp.vec3(0.0, 0.0, 10.0), wp.quat_identity())
        builder = newton.ModelBuilder(up_axis=prototype.up_axis)
        source_ids: list[int] = []
        target_ids: list[list[int]] = []
        for candidate in range(self.capacity):
            builder.begin_world(label=f"reset_candidate_{candidate}")
            builder.add_builder(prototype)
            source_ids.append(
                builder.add_shape_box(
                    -1,
                    xform=parked,
                    hx=source_half[0],
                    hy=source_half[1],
                    hz=source_half[2],
                    cfg=shape_cfg,
                    label="ResetCandidate/Source",
                )
            )
            world_target_ids = []
            for box_index, (_, half) in enumerate(target_boxes):
                world_target_ids.append(
                    builder.add_shape_box(
                        -1,
                        xform=parked,
                        hx=float(half[0]),
                        hy=float(half[1]),
                        hz=float(half[2]),
                        cfg=shape_cfg,
                        label=f"ResetCandidate/Target/{box_index}",
                    )
                )
            target_ids.append(world_target_ids)
            builder.add_shape_plane(
                -1,
                xform=wp.transform_identity(),
                width=0.0,
                length=0.0,
                cfg=shape_cfg,
                label="ResetCandidate/Table",
            )
            builder.end_world()

        self.model = builder.finalize(device=device)
        if self.model.world_count != self.capacity:
            raise RuntimeError("Collision screen built an unexpected number of candidate worlds.")
        self.state = self.model.state()
        self._joint_q = wp.to_torch(self.model.joint_q).reshape(self.capacity, self.coordinate_count)
        self._joint_q_default = self._joint_q.clone()
        self._shape_transform = wp.to_torch(self.model.shape_transform)
        self._shape_transform_default = self._shape_transform.clone()
        self._source_ids = torch.tensor(source_ids, device=device, dtype=torch.long)
        self._target_ids = torch.tensor(target_ids, device=device, dtype=torch.long)

        body_labels = [str(label) for label in self.model.body_label]
        body_names = [label.rsplit("/", 1)[-1] for label in body_labels]
        self._body_robot = torch.tensor(["/Robot/" in label for label in body_labels], device=device, dtype=torch.int32)
        self._body_tests_table = torch.tensor(
            [name in _TABLE_TEST_BODIES for name in body_names], device=device, dtype=torch.int32
        )
        self._body_finger = torch.tensor(
            [name in {"panda_leftfinger", "panda_rightfinger"} for name in body_names],
            device=device,
            dtype=torch.int32,
        )
        self._body_no_source_contact = torch.zeros_like(self._body_finger)

        shape_labels = [str(label) for label in self.model.shape_label]
        explicit = torch.tensor(
            ["ResetCandidate/" in label for label in shape_labels], device=device, dtype=torch.int32
        )
        prototype_table = torch.tensor(
            ["/Table/" in label or label.endswith("/Table") for label in shape_labels],
            device=device,
            dtype=torch.int32,
        )
        self._shape_obstacle = torch.maximum(explicit, prototype_table)
        self._shape_table = torch.tensor(
            [label.endswith("ResetCandidate/Table") for label in shape_labels], device=device, dtype=torch.int32
        )
        self._shape_table = torch.maximum(self._shape_table, prototype_table)
        self._shape_source = torch.tensor(
            [label.endswith("ResetCandidate/Source") for label in shape_labels], device=device, dtype=torch.int32
        )
        if int(self._body_finger.sum()) != 2 * self.capacity or int(self._shape_source.sum()) != self.capacity:
            raise RuntimeError("Collision screen could not identify the replicated Panda fingers and source boxes.")

        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            include_static_kinematic_pairs=False,
            soft_contact_max=0,
            verify_buffers=True,
        )
        self.contacts = self.pipeline.contacts()
        self._colliding_worlds = wp.zeros(self.capacity, dtype=wp.int32, device=self.model.device)

    def check(
        self,
        robot_q: torch.Tensor,
        source_position: torch.Tensor,
        source_quaternion: torch.Tensor,
        target_position: torch.Tensor,
        *,
        allow_source_finger_contact: bool,
    ) -> torch.Tensor:
        count = robot_q.shape[0]
        if count == 0:
            return torch.empty(0, device=robot_q.device, dtype=torch.bool)
        if not 0 < count <= self.capacity:
            raise ValueError("Collision-screen batch exceeds its configured capacity.")

        self._joint_q.copy_(self._joint_q_default)
        self._joint_q[:count].copy_(robot_q)
        self._shape_transform.copy_(self._shape_transform_default)
        center_offset = torch.zeros_like(source_position)
        center_offset[:, 2] = self._source_half_height
        source_center = source_position + math_utils.quat_apply(source_quaternion, center_offset)
        self._shape_transform.index_copy_(
            0, self._source_ids[:count], torch.cat((source_center, source_quaternion), dim=-1)
        )
        target_centers = target_position[:, None, :] + self._target_offsets[None, :, :]
        target_quaternion = torch.zeros((count, len(self._target_offsets), 4), device=robot_q.device)
        target_quaternion[:, :, 3] = 1.0
        self._shape_transform.index_copy_(
            0,
            self._target_ids[:count].reshape(-1),
            torch.cat((target_centers, target_quaternion), dim=-1).reshape(-1, 7),
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.pipeline.collide(self.state, self.contacts)
        wp.to_torch(self._colliding_worlds).zero_()
        body_allows_source = self._body_finger if allow_source_finger_contact else self._body_no_source_contact
        wp.launch(
            _mark_colliding_worlds,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_max,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.model.shape_world,
                wp.from_torch(self._shape_obstacle, dtype=wp.int32),
                wp.from_torch(self._shape_table, dtype=wp.int32),
                self.model.body_world,
                self.state.body_q,
                wp.from_torch(self._body_robot, dtype=wp.int32),
                wp.from_torch(self._body_tests_table, dtype=wp.int32),
                wp.from_torch(body_allows_source, dtype=wp.int32),
                wp.from_torch(self._shape_source, dtype=wp.int32),
                self.model.shape_margin,
                self.penetration_tolerance,
                self.source_finger_penetration_tolerance,
            ],
            outputs=[self._colliding_worlds],
            device=self.model.device,
        )
        wp.synchronize_device(self.model.device)
        if int(wp.to_torch(self.contacts.rigid_contact_count)[0]) > self.contacts.rigid_contact_max:
            raise RuntimeError("Collision-screen contact capacity was exceeded.")
        return (wp.to_torch(self._colliding_worlds)[:count] == 0).clone()


class _Generator:
    """Create a side-balanced connected reset distribution."""

    def __init__(self, env: _GeneratorEnv, cfg: GeneratorCfg):
        self.env = env
        self.task_cfg = env.cfg
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.random = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.attempt_counts = {_NON_GRASPING: 0, _GRASPING: 0}
        self.rejection_counts: dict[int, dict[str, int]] = {
            _NON_GRASPING: defaultdict(int),
            _GRASPING: defaultdict(int),
        }
        self._build_context()

    def _build_context(self) -> None:
        from isaaclab_newton.ik import (
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
            NewtonIKSolver,
            NewtonIKSolverCfg,
        )

        import isaaclab.sim as sim_utils
        from isaaclab import cloner

        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, self.env._robot.cfg.prim_path) if plan is not None else None
        if resolved is None:
            raise RuntimeError("Could not resolve the Franka clone-plan source.")
        source_builder = copy_newton_clone_source(resolved[0])
        prototype_origin = -self.env.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*prototype_origin.tolist()), wp.quat_identity())
        self.prototype = newton.ModelBuilder(up_axis=source_builder.up_axis)
        self.prototype.add_builder(source_builder, xform=prototype_xform)
        if not any("/Table/" in str(label) or str(label).endswith("/Table") for label in self.prototype.shape_label):
            table_path = self.env.scene["table"].cfg.prim_path
            table_resolved = cloner.query.path_to_source(plan, table_path)
            if table_resolved is None:
                raise RuntimeError("Could not resolve the SeattleLab table clone-plan source.")
            self.prototype.add_builder(copy_newton_clone_source(table_resolved[0]), xform=prototype_xform)
        if not any("/Table/" in str(label) or str(label).endswith("/Table") for label in self.prototype.shape_label):
            raise RuntimeError("The reset generator requires the SeattleLab table collision geometry.")
        self.support_lower, self.support_upper = _tabletop_bounds(self.env, resolved[0])
        self.ik_model = self.prototype.finalize(device=str(self.device))

        body_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.body_label]
        hand_matches = [index for index, name in enumerate(body_names) if name == self.task_cfg.tcp_body_name]
        if len(hand_matches) != 1:
            raise RuntimeError(f"Expected one IK body named {self.task_cfg.tcp_body_name!r}.")
        self.hand_id = hand_matches[0]
        joint_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.joint_label]
        joint_q_start = wp.to_torch(self.ik_model.joint_q_start).to(device=self.device, dtype=torch.long)

        def coordinate_id(name: str) -> int:
            matches = [index for index, joint_name in enumerate(joint_names) if joint_name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one IK joint named {name!r}.")
            return int(joint_q_start[matches[0]].item())

        self.arm_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in _ARM_JOINT_NAMES], device=self.device, dtype=torch.long
        )
        self.finger_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in _FINGER_JOINT_NAMES], device=self.device, dtype=torch.long
        )
        self.arm_limits = self.env._joint_pos_limits_t[0, self.env._arm_joint_ids].to(self.device)
        finger_limits = self.env._joint_pos_limits_t[0, self.env._finger_joint_ids].to(self.device)
        self.finger_range = (float(finger_limits[:, 0].max()), float(finger_limits[:, 1].min()))
        arm_home = tuple(self.task_cfg.scene.robot.init_state.joint_pos[name] for name in _ARM_JOINT_NAMES)
        self.arm_home = torch.tensor(arm_home, device=self.device)
        self.gripper_open_position = float(self.task_cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"])
        self.tcp_offset_position = torch.tensor(self.task_cfg.tcp_offset_pos, device=self.device)
        self.tcp_offset_quaternion = torch.tensor(self.task_cfg.tcp_offset_rot, device=self.device)
        self.base_grasp_quaternion = torch.tensor(CUP_GRASP_TCP_QUAT_C, device=self.device)
        self.joint_seed = wp.to_torch(self.ik_model.joint_q).to(self.device).repeat(self.cfg.batch_size, 1)
        self.joint_seed[:, self.arm_coordinate_ids] = self.arm_home

        objective_name = "reset_dataset_tcp"
        objectives = [
            NewtonIKPoseObjectiveCfg(
                body_name=self.task_cfg.tcp_body_name,
                name=objective_name,
                body_offset_pos=self.task_cfg.tcp_offset_pos,
                body_offset_rot=self.task_cfg.tcp_offset_rot,
                position_weight=100.0,
                rotation_weight=5.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=1.0),
        ]
        self.ik_solver = NewtonIKSolver(
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
            model=self.ik_model,
            num_envs=self.cfg.batch_size,
            device=str(self.device),
            objectives=objectives,
            link_resolver=lambda _name: self.hand_id,
        )
        self.pose_objective = self.ik_solver.objectives_by_name[objective_name]

        self.source_half = torch.tensor(SOURCE_CUP_GEOMETRY.outer_half_extents, device=self.device)
        self.cup_center_offset = torch.tensor((0.0, 0.0, float(self.source_half[2])), device=self.device)
        self.cup_grasp_offset = torch.tensor((0.0, 0.0, CUP_GRASP_HEIGHT), device=self.device)
        self.source_mouth_offset = torch.tensor((0.0, 0.0, SOURCE_CUP_GEOMETRY.rim_height), device=self.device)
        self.source_mouth_from_center_height = float(self.source_mouth_offset[2] - self.cup_center_offset[2])
        nominal_source = torch.tensor(self.task_cfg.scene.source_cup.init_state.pos, device=self.device)
        self.source_region_center = nominal_source + self.cup_center_offset
        self.target_half = torch.tensor(TARGET_CUP_GEOMETRY.outer_half_extents, device=self.device)
        self.target_rim_height = 2.0 * float(self.target_half[2])

        default_particle_state = self.env._media.data.default_particle_state_w.torch[0, :, :3]
        self.local_particles = (default_particle_state - self.env.env_origins[0] - nominal_source).to(torch.float32)
        source_lower = torch.tensor(self.env._source_inner_lo, device=self.device)
        source_upper = torch.tensor(self.env._source_inner_hi, device=self.device)
        if not bool(((self.local_particles >= source_lower) & (self.local_particles <= source_upper)).all()):
            raise RuntimeError("The generated MPM layout is outside the source-cup cavity.")
        self.collision_screen = _CollisionScreen(
            self.prototype,
            self.cfg,
            self.task_cfg.collider_margin,
            str(self.device),
        )

    @torch.inference_mode()
    def generate(self) -> dict[str, Any]:
        parts = []
        counts = phase_counts(self.cfg)
        for phase in _PHASE_ORDER:
            required = counts[phase]
            category = _PHASES[phase].category
            before = self.attempt_counts[category]
            parts.append(self._sample_phase(category, phase, required))
            attempts = self.attempt_counts[category] - before
            print(f"[RESET GENERATION] {_PHASE_NAMES[phase]}: accepted {required}/{attempts}", flush=True)

        names = set(RESET_DATASET_STATE_NAMES) | {"_objective_raw", "_difficulty_raw", "_phase"}
        states = {name: torch.cat([part[name] for part in parts], dim=0) for name in names}
        grasping = states["category"] == _GRASPING
        raw_objective = states.pop("_objective_raw")
        grasp_raw = raw_objective[grasping]
        span = grasp_raw.max() - grasp_raw.min()
        if float(span) <= torch.finfo(grasp_raw.dtype).eps:
            raise RuntimeError("The generated grasp objective is degenerate.")
        states["objective"][grasping] = (grasp_raw - grasp_raw.min()) / span
        states["difficulty"] = _normalize_progress(states.pop("_difficulty_raw"), states.pop("_phase"))
        permutation = torch.randperm(len(states["category"]), device=self.device, generator=self.random)
        cpu_states = {name: states[name][permutation].detach().cpu().contiguous() for name in RESET_DATASET_STATE_NAMES}

        task_contract = _reset_dataset_task_contract(self.task_cfg)
        metadata = {
            "task_contract": task_contract,
            "generator_cfg": asdict(self.cfg),
            "phase_names": _PHASE_NAMES,
            "phase_counts": torch.tensor(counts, dtype=torch.int64),
            "attempt_counts": torch.tensor(
                (self.attempt_counts[_NON_GRASPING], self.attempt_counts[_GRASPING]), dtype=torch.int64
            ),
            "rejection_counts": {
                reason: torch.tensor(
                    (
                        self.rejection_counts[_NON_GRASPING].get(reason, 0),
                        self.rejection_counts[_GRASPING].get(reason, 0),
                    ),
                    dtype=torch.int64,
                )
                for reason in sorted(set(self.rejection_counts[_NON_GRASPING]) | set(self.rejection_counts[_GRASPING]))
            },
        }
        local = self.local_particles.detach().cpu().contiguous()
        payload: dict[str, Any] = {
            "format": FRANKA_POUR_RESET_DATASET_FORMAT,
            "schema_version": FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
            "metadata": metadata,
            "states": cpu_states,
            "particle_layouts": {
                "local_position": local.unsqueeze(0),
                "local_velocity": torch.zeros_like(local).unsqueeze(0),
            },
        }
        payload["content_sha256"] = reset_dataset_content_digest(payload)
        reset_dataset_validate_runtime(
            payload,
            expected_task_contract=task_contract,
        )
        return payload

    def _sample_phase(self, category: int, phase: int, required: int) -> dict[str, torch.Tensor]:
        accepted: list[dict[str, torch.Tensor]] = []
        accepted_count = 0
        accepted_sides = torch.zeros(4, device=self.device, dtype=torch.long)
        side_quotas = torch.zeros(4, device=self.device, dtype=torch.long)
        per_side, extra = divmod(required, len(self.cfg.grasp_side_ids))
        for index, side in enumerate(self.cfg.grasp_side_ids):
            side_quotas[side] = per_side + int(index < extra)
        attempted = 0
        maximum = required * self.cfg.max_attempt_multiplier
        while accepted_count < required and attempted < maximum:
            count = min(self.cfg.batch_size, maximum - attempted)
            proposal = self._propose(category, phase, accepted_sides, side_quotas, count)
            attempted += count
            self.attempt_counts[category] += count
            valid = self._validate(proposal, category)
            keep = torch.zeros_like(valid)
            for side in self.cfg.grasp_side_ids:
                rows = torch.where(valid & (proposal["_grasp_side"] == side))[0]
                remaining = int(side_quotas[side] - accepted_sides[side])
                chosen = rows[: max(remaining, 0)]
                keep[chosen] = True
                accepted_sides[side] += chosen.numel()
            self.rejection_counts[category]["quota_full"] += int((valid & ~keep).sum())
            if bool(keep.any()):
                accepted.append(
                    {
                        key: value[keep].detach().clone()
                        for key, value in proposal.items()
                        if key in RESET_DATASET_STATE_NAMES or key in {"_objective_raw", "_difficulty_raw", "_phase"}
                    }
                )
                accepted_count += int(keep.sum())
        if accepted_count != required:
            raise RuntimeError(
                f"Sampling {_PHASE_NAMES[phase]} exhausted {attempted} candidates after accepting "
                f"{accepted_count}/{required}; rejections={dict(self.rejection_counts[category])}."
            )
        return {key: torch.cat([part[key] for part in accepted], dim=0) for key in accepted[0]}

    def _propose(
        self,
        category: int,
        phase: int,
        accepted_sides: torch.Tensor,
        side_quotas: torch.Tensor,
        count: int,
    ) -> dict[str, torch.Tensor]:
        spec = _PHASES[phase]
        if category != spec.category:
            raise ValueError(f"Phase {spec.name!r} does not belong to category {category}.")
        source_position, source_quaternion, target_position = self._sample_table_pair(count)
        target_quaternion = self._identity_quaternions(count)
        finger_position = torch.empty((count, 2), device=self.device)

        deficit = side_quotas - accepted_sides
        active_sides = torch.where(deficit > 0)[0]
        side_order = active_sides[torch.argsort(deficit[active_sides], descending=True)]
        grasp_side = side_order[torch.arange(count, device=self.device) % len(side_order)].to(torch.int8)
        side_angle = grasp_side.float() * (0.5 * math.pi)
        side_axis = torch.zeros((count, 3), device=self.device)
        side_axis[:, 2] = 1.0
        side_rotation = math_utils.quat_from_angle_axis(side_angle, side_axis)
        base_grasp = self.base_grasp_quaternion.expand(count, -1)

        if category == _GRASPING:
            cup_to_tcp = math_utils.quat_mul(side_rotation, base_grasp)
            position_std = (
                self.cfg.near_goal_grasp_position_std if spec.region == _NEAR_GOAL else self.cfg.grasp_position_std
            )
            tcp_seating_offset = torch.randn((count, 3), device=self.device, generator=self.random)
            tcp_seating_offset *= torch.tensor(position_std, device=self.device)
            if spec.region == _NEAR_GOAL:
                source_quaternion, cup_center = self._sample_near_goal(target_position, phase)
            else:
                source_quaternion, cup_center = self._sample_transport(
                    source_position, source_quaternion, target_position, phase
                )
            tcp_quaternion = math_utils.quat_unique(math_utils.quat_mul(source_quaternion, cup_to_tcp))
            source_position = cup_center - math_utils.quat_apply(
                source_quaternion, self.cup_center_offset.expand(count, -1)
            )
            grasp_position = source_position + math_utils.quat_apply(
                source_quaternion, self.cup_grasp_offset.expand(count, -1)
            )
            tcp_position = grasp_position - math_utils.quat_apply(tcp_quaternion, tcp_seating_offset)
            finger_position.fill_(float(self.source_half[1]))
            finger_target = torch.full_like(finger_position, float(self.task_cfg.actions.gripper_action.close_position))
        else:
            radius_attr, cone_attr, rotation_attr = _TCP_SAMPLING[phase]
            radius_range = getattr(self.cfg, radius_attr)
            cone_angle = 0.0 if cone_attr is None else getattr(self.cfg, cone_attr)
            rotation_error = 0.0 if rotation_attr is None else getattr(self.cfg, rotation_attr)
            tcp_position, tcp_quaternion = self._sample_near_object_tcp(
                source_position,
                source_quaternion,
                grasp_side,
                radius_range,
                cone_angle,
                rotation_error,
            )
            finger_mode = _FINGER_MODES.get(phase, "open")
            if finger_mode == "contact":
                finger_position.fill_(
                    min(
                        float(self.source_half[1]) + self.cfg.contact_finger_clearance,
                        self.gripper_open_position,
                    )
                )
                finger_target = torch.full_like(
                    finger_position, float(self.task_cfg.actions.gripper_action.close_position)
                )
            elif finger_mode == "capture":
                gripper_cfg = self.task_cfg.actions.gripper_action
                steps = torch.randint(
                    self.cfg.capture_filter_step_range[0],
                    self.cfg.capture_filter_step_range[1] + 1,
                    (count,),
                    device=self.device,
                    generator=self.random,
                )
                close = float(gripper_cfg.close_position)
                opened = self.gripper_open_position
                filtered = close + (opened - close) * torch.pow(
                    finger_position.new_tensor(1.0 - float(gripper_cfg.alpha)), steps
                )
                filtered.clamp_(
                    min=min(float(self.source_half[1]) + self.cfg.contact_finger_clearance, opened),
                    max=opened,
                )
                finger_position.copy_(filtered[:, None].expand(-1, 2))
                finger_target = finger_position.clone()
            else:
                finger_position.fill_(self.gripper_open_position)
                finger_target = finger_position.clone()
            finger_position[:, 1] = finger_position[:, 0]

        robot_q, actual_tcp_position, actual_tcp_quaternion, ik_valid = self._solve_ik(
            tcp_position, tcp_quaternion, finger_position
        )
        source_pose = torch.cat((source_position, source_quaternion), dim=-1)
        target_pose = torch.cat((target_position, target_quaternion), dim=-1)
        objective_raw = torch.full((count,), -1.0, device=self.device)
        if category == _GRASPING:
            objective_raw = _grasp_objective(
                source_pose,
                target_pose,
                self.source_region_center,
                self.cup_center_offset,
                self.target_rim_height,
                self.cfg,
            )
        objective = objective_raw.clone()
        difficulty_raw = self._difficulty(
            phase,
            source_pose,
            target_pose,
            actual_tcp_position,
            actual_tcp_quaternion,
            objective_raw,
            grasp_side,
            finger_position,
        )
        return {
            "arm_joint_position": robot_q[:, self.arm_coordinate_ids],
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
            "reset_region": torch.full((count,), spec.region, device=self.device, dtype=torch.int8),
            "difficulty": difficulty_raw.clone(),
            "particle_layout_id": torch.zeros(count, device=self.device, dtype=torch.int32),
            "_objective_raw": objective_raw,
            "_difficulty_raw": difficulty_raw,
            "_phase": torch.full((count,), phase, device=self.device, dtype=torch.int8),
            "_grasp_side": grasp_side,
            "_robot_q": robot_q,
            "_ik_valid": ik_valid,
            "_tcp_position": actual_tcp_position,
            "_tcp_quaternion": actual_tcp_quaternion,
        }

    def _solve_ik(
        self, tcp_position: torch.Tensor, tcp_quaternion: torch.Tensor, finger_position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(tcp_position)
        if count < self.cfg.batch_size:
            padding = self.cfg.batch_size - count
            tcp_position = torch.cat((tcp_position, tcp_position[-1:].expand(padding, -1)))
            tcp_quaternion = torch.cat((tcp_quaternion, tcp_quaternion[-1:].expand(padding, -1)))
            finger_padded = torch.cat((finger_position, finger_position[-1:].expand(padding, -1)))
        else:
            finger_padded = finger_position
        self.pose_objective.position_objective.set_target_positions(
            wp.from_torch(tcp_position.contiguous(), dtype=wp.vec3)
        )
        self.pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(tcp_quaternion.contiguous(), dtype=wp.vec4)
        )
        seed = self.joint_seed.clone()
        seed[:, self.finger_coordinate_ids] = finger_padded
        self.ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.cfg.batch_size, self.cfg.ik_seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.cfg.batch_size, self.cfg.ik_seeds)
        residuals = wp.to_torch(self.ik_solver.solver.residuals).reshape(self.cfg.batch_size, self.cfg.ik_seeds, -1)
        position_residuals = torch.linalg.vector_norm(residuals[:, :, :3] / 100.0, dim=-1)
        rotation_residuals = torch.linalg.vector_norm(residuals[:, :, 3:6] / 5.0, dim=-1)
        arm_q = joint_q[:, :, self.arm_coordinate_ids]
        arm_margin = torch.minimum(arm_q - self.arm_limits[:, 0], self.arm_limits[:, 1] - arm_q).amin(-1)
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & torch.isfinite(position_residuals)
            & torch.isfinite(rotation_residuals)
            & (costs <= self.cfg.ik_max_cost)
            & (arm_margin >= self.cfg.ik_joint_margin)
            & (position_residuals <= self.cfg.ik_max_position_residual)
            & (rotation_residuals <= self.cfg.ik_max_rotation_residual)
            & (torch.linalg.vector_norm(arm_q - self.arm_home, dim=-1) <= self.cfg.ik_max_home_distance)
        )
        valid_cost = costs.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        best = torch.where(valid, valid_cost.argmin(-1), fallback)
        rows = torch.arange(count, device=self.device)
        selected = best[rows]
        solved = joint_q[rows, selected].clone()
        solved[:, self.finger_coordinate_ids] = finger_position
        body_poses = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.cfg.batch_size, self.cfg.ik_seeds, -1, 7)
        hand_pose = body_poses[rows, selected, self.hand_id].clone()
        actual_position, actual_quaternion = math_utils.combine_frame_transforms(
            hand_pose[:, :3],
            hand_pose[:, 3:7],
            self.tcp_offset_position.expand(count, -1),
            self.tcp_offset_quaternion.expand(count, -1),
        )
        return solved, actual_position, actual_quaternion, valid[rows]

    def _validate(self, proposal: dict[str, torch.Tensor], category: int) -> torch.Tensor:
        count = len(proposal["category"])
        phase = int(proposal["_phase"][0])
        spec = _PHASES[phase]
        valid = torch.ones(count, device=self.device, dtype=torch.bool)
        valid = self._reject(valid, proposal["_ik_valid"], category, "invalid_ik")

        source_pose = proposal["source_root_pose"]
        target_pose = proposal["target_root_pose"]
        source_center = box_center_from_base_pose(source_pose, self.source_half)
        target_center = box_center_from_base_pose(target_pose, self.target_half)
        clear = ~oriented_boxes_overlap(
            source_center,
            source_pose[:, 3:7],
            self.source_half,
            target_center,
            target_pose[:, 3:7],
            self.target_half,
            self.cfg.obstacle_clearance,
        )
        valid = self._reject(valid, clear, category, "source_target_collision")
        valid = self._reject(
            valid,
            box_above_plane(
                source_pose,
                self.source_half,
                penetration_tolerance=self.cfg.collision_penetration_tolerance,
            ),
            category,
            "source_table_collision",
        )
        valid = self._reject(
            valid,
            box_above_plane(
                target_pose,
                self.target_half,
                penetration_tolerance=self.cfg.collision_penetration_tolerance,
            ),
            category,
            "target_table_collision",
        )
        valid = self._reject(
            valid,
            box_supported(
                target_pose,
                self.target_half,
                self.support_lower,
                self.support_upper,
                self.cfg.obstacle_clearance,
            ),
            category,
            "target_table_support",
        )
        if category == _NON_GRASPING:
            valid = self._reject(
                valid,
                box_supported(
                    source_pose,
                    self.source_half,
                    self.support_lower,
                    self.support_upper,
                    self.cfg.obstacle_clearance,
                ),
                category,
                "source_table_support",
            )
        valid = self._reject(valid, self._particles_in_workspace(source_pose), category, "particle_workspace")

        if spec.region == _NEAR_GOAL:
            mouth = source_pose[:, :3] + math_utils.quat_apply(
                source_pose[:, 3:7], self.source_mouth_offset.expand(count, -1)
            )
            horizontal = torch.linalg.vector_norm(mouth[:, :2] - target_pose[:, :2], dim=-1)
            clearance = mouth[:, 2] - (target_pose[:, 2] + self.target_rim_height)
            local_up = torch.zeros((count, 3), device=self.device)
            local_up[:, 2] = 1.0
            world_up = math_utils.quat_apply(source_pose[:, 3:7], local_up)
            near_goal_valid = (
                (horizontal <= self.cfg.near_pour_horizontal_radius)
                & (clearance >= self.cfg.near_pour_height_range[0])
                & (world_up[:, 2] <= math.cos(self.cfg.boundary_tilt_range[0]))
            )
            valid = self._reject(valid, near_goal_valid, category, "invalid_near_pour_geometry")

        grasp_point = source_pose[:, :3] + math_utils.quat_apply(
            source_pose[:, 3:7], self.cup_grasp_offset.expand(count, -1)
        )
        if category == _GRASPING:
            cup_offset_tcp = math_utils.quat_apply_inverse(
                proposal["_tcp_quaternion"], grasp_point - proposal["_tcp_position"]
            )
            side_angle = proposal["_grasp_side"].float() * (0.5 * math.pi)
            side_axis = torch.zeros((count, 3), device=self.device)
            side_axis[:, 2] = 1.0
            expected = math_utils.quat_mul(
                math_utils.quat_from_angle_axis(side_angle, side_axis),
                self.base_grasp_quaternion.expand(count, -1),
            )
            actual = math_utils.quat_mul(math_utils.quat_conjugate(source_pose[:, 3:7]), proposal["_tcp_quaternion"])
            seating_valid = (
                cup_offset_tcp.abs() <= torch.tensor(self.cfg.grasp_seating_max_offset, device=self.device)
            ).all(-1) & (math_utils.quat_error_magnitude(actual, expected) <= self.cfg.grasp_seating_max_rotation_error)
            valid = self._reject(valid, seating_valid, category, "invalid_grasp_seating")
        else:
            tcp_distance = torch.linalg.vector_norm(grasp_point - proposal["_tcp_position"], dim=-1)
            if phase != _REACHING_BROAD:
                radius_range = getattr(self.cfg, _TCP_SAMPLING[phase][0])
                valid = self._reject(
                    valid,
                    (tcp_distance >= radius_range[0]) & (tcp_distance <= radius_range[1]),
                    category,
                    "reaching_aligned_tcp_shell" if phase == _REACHING_ALIGNED else "near_object_tcp_shell",
                )
                if phase == _CAPTURE:
                    offset = math_utils.quat_apply_inverse(source_pose[:, 3:7], proposal["_tcp_position"] - grasp_point)
                    direction = torch.nn.functional.normalize(offset, dim=-1)
                    side_angle = proposal["_grasp_side"].float() * (0.5 * math.pi)
                    side_axis = torch.zeros((count, 3), device=self.device)
                    side_axis[:, 2] = 1.0
                    side_rotation = math_utils.quat_from_angle_axis(side_angle, side_axis)
                    nominal = torch.zeros((count, 3), device=self.device)
                    nominal[:, 0] = -1.0
                    nominal = math_utils.quat_apply(side_rotation, nominal)
                    valid = self._reject(
                        valid,
                        (direction * nominal).sum(-1) >= math.cos(self.cfg.capture_cone_angle),
                        category,
                        "near_object_capture_cone",
                    )
                    expected = math_utils.quat_mul(
                        side_rotation,
                        self.base_grasp_quaternion.expand(count, -1),
                    )
                    actual = math_utils.quat_mul(
                        math_utils.quat_conjugate(source_pose[:, 3:7]), proposal["_tcp_quaternion"]
                    )
                    valid = self._reject(
                        valid,
                        math_utils.quat_error_magnitude(actual, expected) <= self.cfg.capture_rotation_error,
                        category,
                        "near_object_capture_rotation",
                    )
            else:
                center_distance = torch.linalg.vector_norm(source_center - proposal["_tcp_position"], dim=-1)
                valid = self._reject(
                    valid,
                    center_distance >= self.cfg.non_grasping_min_tcp_source_distance,
                    category,
                    "source_inside_gripper",
                )

        if category == _GRASPING:
            indices = torch.where(valid)[0]
            if indices.numel():
                parked_source = source_pose[indices, :3].clone()
                parked_source[:, 2] = 10.0
                clear = self.collision_screen.check(
                    proposal["_robot_q"][indices],
                    parked_source,
                    source_pose[indices, 3:7],
                    target_pose[indices, :3],
                    allow_source_finger_contact=False,
                )
                full = torch.zeros_like(valid)
                full[indices] = clear
                valid = self._reject(valid, full, category, "robot_preinsert_collision")

        indices = torch.where(valid)[0]
        if indices.numel():
            clear = self.collision_screen.check(
                proposal["_robot_q"][indices],
                source_pose[indices, :3],
                source_pose[indices, 3:7],
                target_pose[indices, :3],
                allow_source_finger_contact=category == _GRASPING or phase in _FINGER_MODES,
            )
            full = torch.zeros_like(valid)
            full[indices] = clear
            valid = self._reject(valid, full, category, "complete_state_collision")
        return valid

    def _particles_in_workspace(self, source_pose: torch.Tensor) -> torch.Tensor:
        count = len(source_pose)
        particle_count = len(self.local_particles)
        quaternion = source_pose[:, None, 3:7].expand(-1, particle_count, -1).reshape(-1, 4)
        local = self.local_particles.expand(count, -1, -1).reshape(-1, 3)
        world = source_pose[:, None, :3] + math_utils.quat_apply(quaternion, local).reshape(count, particle_count, 3)
        lower = torch.tensor(self.task_cfg.particle_workspace_lower_bound, device=self.device)
        upper = torch.tensor(self.task_cfg.particle_workspace_upper_bound, device=self.device)
        return ((world >= lower) & (world <= upper)).all(-1).all(-1)

    def _reject(self, current: torch.Tensor, check: torch.Tensor, category: int, reason: str) -> torch.Tensor:
        self.rejection_counts[category][reason] += int((current & ~check).sum())
        return current & check

    def _sample_near_object_tcp(
        self,
        source_position: torch.Tensor,
        source_quaternion: torch.Tensor,
        grasp_side: torch.Tensor,
        radius_range: tuple[float, float],
        cone_angle: float,
        rotation_error: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(source_position)
        radius = torch.empty(count, device=self.device).uniform_(*radius_range, generator=self.random)
        cosine = torch.empty(count, device=self.device).uniform_(math.cos(cone_angle), 1.0, generator=self.random)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min_(0.0))
        azimuth = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.random)
        side_angle = grasp_side.float() * (0.5 * math.pi)
        side_axis = torch.zeros((count, 3), device=self.device)
        side_axis[:, 2] = 1.0
        side_rotation = math_utils.quat_from_angle_axis(side_angle, side_axis)
        approach = torch.stack((-cosine, sine * torch.cos(azimuth), sine * torch.sin(azimuth)), dim=-1)
        approach = math_utils.quat_apply(side_rotation, approach) * radius[:, None]
        grasp_point = source_position + math_utils.quat_apply(
            source_quaternion, self.cup_grasp_offset.expand(count, -1)
        )
        tcp_position = grasp_point + math_utils.quat_apply(source_quaternion, approach)
        cup_to_tcp = math_utils.quat_mul(
            side_rotation,
            self.base_grasp_quaternion.expand(count, -1),
        )
        aligned = math_utils.quat_mul(source_quaternion, cup_to_tcp)
        noise_axis = torch.nn.functional.normalize(
            torch.randn((count, 3), device=self.device, generator=self.random), dim=-1
        )
        noise_angle = torch.empty(count, device=self.device).uniform_(0.0, rotation_error, generator=self.random)
        noise = math_utils.quat_from_angle_axis(noise_angle, noise_axis)
        return tcp_position, math_utils.quat_unique(math_utils.quat_mul(aligned, noise))

    def _sample_table_source(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        radius = torch.sqrt(
            torch.empty(count, device=self.device).uniform_(
                self.cfg.source_radius_range[0] ** 2,
                self.cfg.source_radius_range[1] ** 2,
                generator=self.random,
            )
        )
        angle = torch.empty(count, device=self.device).uniform_(
            -self.cfg.source_azimuth_range, self.cfg.source_azimuth_range, generator=self.random
        )
        xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
        position = torch.zeros((count, 3), device=self.device)
        position[:, :2] = xy
        yaw = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.random)
        quaternion = torch.zeros((count, 4), device=self.device)
        quaternion[:, 2] = torch.sin(0.5 * yaw)
        quaternion[:, 3] = torch.cos(0.5 * yaw)
        return position, quaternion

    def _sample_table_pair(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_position, source_quaternion = self._sample_table_source(count)
        supported_lower = self.support_lower + self.target_half[:2] + self.cfg.obstacle_clearance
        supported_upper = self.support_upper - self.target_half[:2] - self.cfg.obstacle_clearance
        target_position = torch.zeros_like(source_position)
        unresolved = torch.ones(count, device=self.device, dtype=torch.bool)
        for _ in range(64):
            rows = torch.where(unresolved)[0]
            if rows.numel() == 0:
                break
            radius = torch.sqrt(
                torch.empty(len(rows), device=self.device).uniform_(
                    self.cfg.source_target_distance_range[0] ** 2,
                    self.cfg.source_target_distance_range[1] ** 2,
                    generator=self.random,
                )
            )
            angle = torch.empty(len(rows), device=self.device).uniform_(-math.pi, math.pi, generator=self.random)
            xy = source_position[rows, :2] + torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
            supported = ((xy >= supported_lower) & (xy <= supported_upper)).all(-1)
            accepted = rows[supported]
            target_position[accepted, :2] = xy[supported]
            unresolved[accepted] = False
        if bool(unresolved.any()):
            raise RuntimeError("Could not sample a supported source/receiver pair.")
        return source_position, source_quaternion, target_position

    def _sample_transport(
        self,
        pickup_position: torch.Tensor,
        pickup_quaternion: torch.Tensor,
        target_position: torch.Tensor,
        phase: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode, radius_range, height_range, tilt_range, xy_jitter = _TRANSPORT_SAMPLING[phase]
        count = len(target_position)
        pickup_center = pickup_position + math_utils.quat_apply(
            pickup_quaternion, self.cup_center_offset.expand(count, -1)
        )
        cup_center = pickup_center.clone()
        if mode == "pickup":
            cup_center[:, 2] += torch.empty(count, device=self.device).uniform_(*height_range, generator=self.random)
            cup_center[:, :2] += torch.empty((count, 2), device=self.device).uniform_(
                -xy_jitter, xy_jitter, generator=self.random
            )
        elif mode == "interpolate":
            progress = torch.rand(count, device=self.device, generator=self.random)
            pour_xy = target_position[:, :2] + torch.tensor(self.cfg.pour_source_offset_xy, device=self.device)
            cup_center[:, :2] = torch.lerp(pickup_center[:, :2], pour_xy, progress[:, None])
            cup_center[:, :2] += torch.empty((count, 2), device=self.device).uniform_(
                -xy_jitter, xy_jitter, generator=self.random
            )
            cup_center[:, 2] += torch.empty(count, device=self.device).uniform_(*height_range, generator=self.random)
        elif mode == "goal":
            radius = torch.sqrt(
                torch.empty(count, device=self.device).uniform_(
                    radius_range[0] ** 2,
                    radius_range[1] ** 2,
                    generator=self.random,
                )
            )
            angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.random)
            cup_center[:, :2] = target_position[:, :2] + torch.tensor(
                self.cfg.pour_source_offset_xy, device=self.device
            )
            cup_center[:, 0] += radius * torch.cos(angle)
            cup_center[:, 1] += radius * torch.sin(angle)
            cup_center[:, 2] = target_position[:, 2] + self.target_rim_height
            cup_center[:, 2] += torch.empty(count, device=self.device).uniform_(*height_range, generator=self.random)
        else:
            raise ValueError(f"Unsupported transport mode {mode!r} for {_PHASE_NAMES[phase]}.")
        tilt = torch.empty(count, device=self.device).uniform_(*tilt_range, generator=self.random)
        yaw = torch.atan2(
            2.0 * pickup_quaternion[:, 2] * pickup_quaternion[:, 3],
            1.0 - 2.0 * pickup_quaternion[:, 2].square(),
        )
        yaw_axis = torch.zeros((count, 3), device=self.device)
        yaw_axis[:, 2] = 1.0
        tilt_axis = torch.zeros_like(yaw_axis)
        tilt_axis[:, 0] = 1.0
        quaternion = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(yaw, yaw_axis),
            math_utils.quat_from_angle_axis(tilt, tilt_axis),
        )
        return math_utils.quat_unique(quaternion), cup_center

    def _sample_near_goal(self, target_position: torch.Tensor, phase: int) -> tuple[torch.Tensor, torch.Tensor]:
        mode, radius_range, tilt_fractions, radius_relative = _NEAR_GOAL_SAMPLING[phase]
        count = len(target_position)
        azimuth = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.random)
        if mode == "boundary":
            progress = torch.rand(count, device=self.device, generator=self.random)
            tilt = self.cfg.boundary_tilt_range[0] + progress * (
                self.cfg.boundary_tilt_range[1] - self.cfg.boundary_tilt_range[0]
            )
            height_center = self.cfg.near_pour_height_range[1] + progress * (
                self.cfg.near_pour_height_range[0] - self.cfg.near_pour_height_range[1]
            )
            height_jitter = 0.15 * (self.cfg.near_pour_height_range[1] - self.cfg.near_pour_height_range[0])
            clearance = (
                height_center
                + torch.empty(count, device=self.device).uniform_(-height_jitter, height_jitter, generator=self.random)
            ).clamp(*self.cfg.near_pour_height_range)
            radius_limit = 0.5 * self.cfg.near_pour_horizontal_radius + progress * (
                0.005 - 0.5 * self.cfg.near_pour_horizontal_radius
            )
            radial_distance = torch.sqrt(torch.rand(count, device=self.device, generator=self.random)) * radius_limit
        else:
            tilt_span = self.cfg.near_pour_tilt_range[1] - self.cfg.near_pour_tilt_range[0]
            tilt_range = tuple(self.cfg.near_pour_tilt_range[0] + fraction * tilt_span for fraction in tilt_fractions)
            if radius_relative:
                radius_range = tuple(value * self.cfg.near_pour_horizontal_radius for value in radius_range)
            tilt = torch.empty(count, device=self.device).uniform_(*tilt_range, generator=self.random)
            radial_distance = torch.sqrt(
                torch.empty(count, device=self.device).uniform_(
                    radius_range[0] ** 2, radius_range[1] ** 2, generator=self.random
                )
            )
            clearance = torch.empty(count, device=self.device).uniform_(
                *self.cfg.near_pour_height_range, generator=self.random
            )
        tilt_axis = torch.stack((torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)), dim=-1)
        quaternion = math_utils.quat_unique(math_utils.quat_from_angle_axis(tilt, tilt_axis))
        radial_angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi, generator=self.random)
        mouth_from_center_local = torch.zeros((count, 3), device=self.device)
        mouth_from_center_local[:, 2] = self.source_mouth_from_center_height
        mouth_from_center = math_utils.quat_apply(quaternion, mouth_from_center_local)
        cup_center = target_position - mouth_from_center
        cup_center[:, 0] += radial_distance * torch.cos(radial_angle)
        cup_center[:, 1] += radial_distance * torch.sin(radial_angle)
        cup_center[:, 2] += self.target_rim_height + clearance
        return quaternion, cup_center

    def _difficulty(
        self,
        phase: int,
        source_pose: torch.Tensor,
        target_pose: torch.Tensor,
        tcp_position: torch.Tensor,
        tcp_quaternion: torch.Tensor,
        objective: torch.Tensor,
        grasp_side: torch.Tensor,
        finger_position: torch.Tensor,
    ) -> torch.Tensor:
        count = len(source_pose)
        grasp_point = source_pose[:, :3] + math_utils.quat_apply(
            source_pose[:, 3:7], self.cup_grasp_offset.expand(count, -1)
        )
        tcp_distance = torch.linalg.vector_norm(tcp_position - grasp_point, dim=-1)
        if phase in (_REACHING_BROAD, _REACHING_ALIGNED):
            return tcp_distance + (1.0 if phase == _REACHING_BROAD else 0.0)
        if phase in (_CONTACT, _CAPTURE, _PREGRASP, _APPROACH):
            side_angle = grasp_side.float() * (0.5 * math.pi)
            side_axis = torch.zeros((count, 3), device=self.device)
            side_axis[:, 2] = 1.0
            expected = math_utils.quat_mul(
                source_pose[:, 3:7],
                math_utils.quat_mul(
                    math_utils.quat_from_angle_axis(side_angle, side_axis),
                    self.base_grasp_quaternion.expand(count, -1),
                ),
            )
            rotation_error = math_utils.quat_error_magnitude(tcp_quaternion, expected)
            radius_score = ((tcp_distance - 0.001) / (0.08 - 0.001)).clamp(0.0, 1.0)
            phase_rank = {_CONTACT: 0.0, _CAPTURE: 1.0, _PREGRASP: 2.0, _APPROACH: 3.0}[phase]
            opening = finger_position.mean(-1) / max(self.gripper_open_position, 1.0e-6)
            return (
                phase_rank
                + 0.6 * radius_score
                + 0.2 * (rotation_error / math.radians(10.0)).clamp(0.0, 1.0)
                + 0.2 * opening
            )
        if phase in (_BOUNDARY, _QUICK, _IMMEDIATE):
            phase_rank = {_IMMEDIATE: 0.0, _QUICK: 1.0, _BOUNDARY: 0.0}[phase]
            return phase_rank + (1.0 - objective).clamp(0.0, 1.0)

        cup_center = source_pose[:, :3] + math_utils.quat_apply(
            source_pose[:, 3:7], self.cup_center_offset.expand(count, -1)
        )
        goal_center = target_pose[:, :3].clone()
        goal_center[:, :2] += torch.tensor(self.cfg.pour_source_offset_xy, device=self.device)
        goal_center[:, 2] += self.target_rim_height + self.cfg.objective_target_height_threshold
        position_score = (torch.linalg.vector_norm(cup_center - goal_center, dim=-1) / 0.35).clamp(0.0, 1.0)
        local_up = torch.zeros((count, 3), device=self.device)
        local_up[:, 2] = 1.0
        tilt_progress = ((1.0 - math_utils.quat_apply(source_pose[:, 3:7], local_up)[:, 2]) * 0.5).clamp(0.0, 1.0)
        mismatch = torch.abs(tilt_progress - (1.0 - position_score))
        phase_rank = {_POUR_ENTRY: 0.0, _PRE_POUR: 1.0, _UPRIGHT: 2.0, _LOW_LIFT: 3.0, _JUST_GRASPED: 4.0}[phase]
        return phase_rank + 0.75 * position_score + 0.25 * mismatch

    def _identity_quaternions(self, count: int) -> torch.Tensor:
        quaternion = torch.zeros((count, 4), device=self.device)
        quaternion[:, 3] = 1.0
        return quaternion


def _save_atomic(payload: dict[str, Any], output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_attempt_multiplier", type=int, default=100)
    parser.add_argument("--grasping_count", type=int, default=10_000)
    parser.add_argument("--non_grasping_count", type=int, default=10_000)
    add_launcher_args(parser)
    args = parser.parse_args()

    env_cfg = FrankaPourResetDatasetEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    with launch_simulation(env_cfg, args):
        env = _GeneratorEnv(env_cfg)
        try:
            generator_cfg = GeneratorCfg(
                grasping_count=args.grasping_count,
                non_grasping_count=args.non_grasping_count,
                batch_size=args.batch_size,
                seed=args.seed,
                max_attempt_multiplier=args.max_attempt_multiplier,
            )
            payload = _Generator(env, generator_cfg).generate()
            _save_atomic(payload, args.output)
        finally:
            env.close()

    output = args.output.expanduser().resolve()
    print(f"[INFO] Wrote {len(payload['states']['category'])} reset states to {output}.")
    print(f"[INFO] Content SHA-256: {payload['content_sha256']}")
    print("[INFO] Optional reproducibility pin:")
    print(f"       env.reset_dataset_content_sha256={payload['content_sha256']}")


if __name__ == "__main__":
    main()
