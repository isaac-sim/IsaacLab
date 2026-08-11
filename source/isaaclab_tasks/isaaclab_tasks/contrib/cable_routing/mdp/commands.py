# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal commands for the bimanual cable-routing task."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import isaaclab.envs.mdp as env_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

from ..yam_frames import YAM_CONTACT_FRAME_OFFSET_POS, YAM_CONTACT_FRAME_OFFSET_QUAT
from .cable_geometry import cable_relative_joint_gap
from .events import reset_peg_offsets
from .reset_curves import generate_route_conditioned_cable_poses, validate_route_conditioned_cable_poses
from .reset_replay import CableResetReplay, CableResetReplayCfg, finite_scene_state_rows
from .reset_robot_targets import (
    build_top_down_yam_contact_target_poses,
    finite_reset_target_rows,
    select_nearest_cable_segment_indices,
    select_workspace_aware_cable_contact_indices,
    valid_top_down_yam_target_rows,
)
from .route_metrics import benchmark_local_cable_spans, benchmark_winding_angle, ordered_route_state

_LOGGER = logging.getLogger(__name__)

_STEP_COMPLETED = 0
_STEP_ACTIVE = 1
_STEP_PENDING = 2
_DIRECTION_CW = 0
_DIRECTION_CCW = 1
_DIRECTION_COMPLETED = 2


@dataclass(frozen=True)
class _RouteGoalMarkerData:
    """Dense environment-major transforms for route-goal visualization."""

    step_positions_w: torch.Tensor
    direction_positions_w: torch.Tensor
    direction_orientations_w: torch.Tensor
    arc_positions_w: torch.Tensor
    step_scales: torch.Tensor
    direction_scales: torch.Tensor
    arc_scales: torch.Tensor
    step_marker_indices: torch.Tensor
    direction_marker_indices: torch.Tensor
    arc_marker_indices: torch.Tensor


@dataclass(frozen=True)
class _RobotResetCondition:
    """Robot/cable correspondences that must survive physical settling."""

    assigned: torch.Tensor
    segment_indices: torch.Tensor
    height_offsets: torch.Tensor
    ik_valid: torch.Tensor


def _reset_replay_scratch_env_ids(
    all_env_ids: torch.Tensor,
    pending_local: torch.Tensor,
    attempt: int,
    max_attempts: int,
) -> torch.Tensor:
    """Map pending bank rows onto a rotating set of scratch environments.

    Replay rows are logical destinations, not persistent environment clones.  Keeping a
    rejected row attached to one clone also keeps clone-indexed solver seeds and numerical
    conditions attached to every retry.  Sweep the retries across the available worlds while
    preserving a one-to-one row/environment assignment within each attempt.
    """
    num_envs = len(all_env_ids)
    if num_envs == 0:
        return all_env_ids
    stride = max(1, num_envs // max(max_attempts, 1))
    offset = (attempt * stride) % num_envs
    return all_env_ids[(pending_local + offset) % num_envs]


def _route_goal_marker_data(
    peg_positions_w: torch.Tensor,
    peg_indices: torch.Tensor,
    directions: torch.Tensor,
    valid_steps: torch.Tensor,
    completed_steps: torch.Tensor,
    active_step: torch.Tensor,
    marker_offsets: tuple[float, float, float],
    route_arc_radius: float = 0.030,
    route_arc_samples: int = 16,
) -> _RouteGoalMarkerData:
    """Build state beads, ghost wrap rings, and tangent arrows for a route program.

    The marker height and size decrease with route order. At the positive-X side of a peg,
    clockwise motion is tangent along negative Y and counterclockwise motion is tangent along
    positive Y. Invalid padded steps remain in the dense batch with zero scale so Newton can
    retain its environment-major marker layout.
    """
    marker_z_offset, marker_step_z_spacing, marker_radial_offset = marker_offsets
    max_route_steps = peg_indices.shape[1]
    route_steps = torch.arange(max_route_steps, device=peg_indices.device)
    route_steps_batched = route_steps.unsqueeze(0).expand_as(peg_indices)

    selected_peg_positions = torch.gather(
        peg_positions_w,
        1,
        peg_indices[..., None].expand(-1, -1, 3),
    )
    step_positions = selected_peg_positions.clone()
    step_positions[..., 2] += marker_z_offset + marker_step_z_spacing * route_steps
    direction_positions = step_positions.clone()
    direction_positions[..., 0] += marker_radial_offset

    is_active = valid_steps & (route_steps_batched == active_step[:, None])
    step_marker_indices = torch.full_like(peg_indices, _STEP_PENDING)
    step_marker_indices = torch.where(is_active, _STEP_ACTIVE, step_marker_indices)
    step_marker_indices = torch.where(completed_steps, _STEP_COMPLETED, step_marker_indices)

    direction_marker_indices = torch.where(
        directions > 0.0,
        torch.full_like(peg_indices, _DIRECTION_CW),
        torch.full_like(peg_indices, _DIRECTION_CCW),
    )
    direction_marker_indices = torch.where(
        completed_steps,
        torch.full_like(peg_indices, _DIRECTION_COMPLETED),
        direction_marker_indices,
    )

    arc_angles = torch.arange(route_arc_samples, device=directions.device, dtype=directions.dtype) * (
        2.0 * math.pi / route_arc_samples
    )
    arc_positions = selected_peg_positions[:, :, None, :].expand(-1, -1, route_arc_samples, -1).clone()
    arc_positions[..., 0] += route_arc_radius * torch.cos(arc_angles)
    arc_positions[..., 1] += route_arc_radius * torch.sin(arc_angles)
    arc_positions[..., 2] += 0.55 * marker_z_offset + marker_step_z_spacing * route_steps[:, None]

    # Earlier route steps are slightly larger, while the active step is strongly highlighted.
    order_scale = (1.0 - 0.12 * route_steps).clamp(min=0.5).unsqueeze(0)
    step_scale = order_scale.expand_as(directions)
    step_scale = torch.where(is_active, 1.45 * step_scale, step_scale)
    step_scale = torch.where(completed_steps, 0.75 * step_scale, step_scale)
    step_scale = torch.where(valid_steps, step_scale, 0.0)

    direction_scale = 0.78 * order_scale.expand_as(directions)
    direction_scale = torch.where(is_active, 1.25 * order_scale, direction_scale)
    direction_scale = torch.where(completed_steps, 0.65 * order_scale, direction_scale)
    direction_scale = torch.where(valid_steps, direction_scale, 0.0)
    arc_scale = 0.72 * order_scale.expand_as(directions)
    arc_scale = torch.where(is_active, order_scale, arc_scale)
    arc_scale = torch.where(completed_steps, 0.60 * order_scale, arc_scale)
    arc_scale = torch.where(valid_steps, arc_scale, 0.0)

    # The arrow asset points along local +X. Rotate it +/- 90 degrees about +Z at the peg's +X side.
    half_yaw = -0.25 * math.pi * directions
    direction_orientations = directions.new_zeros((*directions.shape, 4))
    direction_orientations[..., 2] = torch.sin(half_yaw)
    direction_orientations[..., 3] = torch.cos(half_yaw)

    return _RouteGoalMarkerData(
        step_positions_w=step_positions.flatten(0, 1),
        direction_positions_w=direction_positions.flatten(0, 1),
        direction_orientations_w=direction_orientations.flatten(0, 1),
        arc_positions_w=arc_positions.flatten(0, 2),
        step_scales=step_scale[..., None].expand(-1, -1, 3).flatten(0, 1),
        direction_scales=direction_scale[..., None].expand(-1, -1, 3).flatten(0, 1),
        arc_scales=arc_scale[..., None, None].expand(-1, -1, route_arc_samples, 3).flatten(0, 2),
        step_marker_indices=step_marker_indices.flatten(),
        direction_marker_indices=direction_marker_indices.flatten(),
        arc_marker_indices=direction_marker_indices[..., None].expand(-1, -1, route_arc_samples).flatten(),
    )


@configclass
class CableRoutingCommandCfg(CommandTermCfg):
    """Configuration for a padded, geometry-grounded route program.

    A route option is a sequence of ``(peg_index, direction)`` pairs. Direction ``+1`` means
    clockwise and ``-1`` means counterclockwise, matching ManipulationNet.
    """

    class_type: type[CableRoutingCommand] | str = "{DIR}.commands:CableRoutingCommand"

    cable_name: str = "cable"
    """Scene name of the cable object."""

    peg_names: tuple[str, ...] = ("peg_0", "peg_1")
    """Scene names of route pegs in canonical order."""

    route_options: tuple[tuple[tuple[int, int], ...], ...] = (
        ((0, -1),),
        ((1, 1),),
        ((0, -1), (1, 1)),
        ((0, 1),),
        ((1, -1),),
        ((0, 1), (1, 1)),
        ((0, -1), (1, -1)),
    )
    """Seven route goals spanning both pegs and both winding directions.

    The first three IDs retain their original staged-task meanings. The four
    additional programs complete the single-peg direction combinations and
    add all-clockwise and all-counterclockwise two-peg routes.
    """

    allowed_route_ids: tuple[int, ...] = (0, 1, 2)
    """Route options sampled at reset. Restrict this tuple for staged training."""

    max_route_steps: int = 3
    """Number of padded goal tokens.

    Current goals use at most two route steps. The third slot preserves the
    established 18-dimensional goal/checkpoint interface and leaves room for a
    later three-fixture route without changing the policy input shape.
    """

    board_origin_b: tuple[float, float, float] = (0.0, 0.0, 0.77)
    """Board-frame origin relative to each environment origin [m]."""

    goal_position_scale: tuple[float, float, float] = (0.15, 0.20, 0.10)
    """Per-axis scale used to bound encoded peg positions [m]."""

    radial_cutoff: float = 0.05
    """Maximum planar distance from a peg used by route metrics [m]."""

    axial_cutoff: float | None = None
    """Maximum vertical distance from a peg center used by route metrics [m].

    ``None`` retains radial-only behavior for fixtures without a finite axial
    extent. The YAM environment configures this from peg and cable geometry.
    """

    completion_winding: float = 2.6
    """Directed winding required to complete a route step [rad]."""

    maximum_completion_winding: float = 2.0 * math.pi + 0.25
    """Largest directed winding accepted as a single wrap [rad]."""

    maximum_local_cable_length: float = 0.25
    """Maximum cable length allowed inside one target peg neighborhood [m]."""

    settled_cable_bounds_b: tuple[tuple[float, float], tuple[float, float]] = ((-0.55, 0.55), (-0.40, 0.40))
    """Tabletop x/y bounds used to qualify physically settled replay cables [m].

    The benchmark cable is longer than its fixture board and may validly cross
    a board edge onto the surrounding table. Generated curves begin on the
    board to avoid teleport overlap; post-settle containment therefore uses the
    actual tabletop instead of incorrectly rejecting benign overhang.
    """

    reset_replay: CableResetReplayCfg = CableResetReplayCfg()
    """Success-conditioned full-scene reset replay configuration."""

    marker_z_offset: float = 0.040
    """Height of the first ordered-step marker above a peg center [m]."""

    marker_step_z_spacing: float = 0.016
    """Additional height for each later route step [m]."""

    marker_radial_offset: float = 0.035
    """Positive-X offset of each tangent direction arrow from its peg [m]."""

    route_arc_radius: float = 0.030
    """Radius of the ghost cable wrap shown around each target peg [m]."""

    route_arc_samples: int = 16
    """Number of beads used to render one ghost cable wrap."""

    route_step_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cable_route_steps",
        markers={
            "completed": sim_utils.SphereCfg(
                radius=0.009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.20)),
            ),
            "active": sim_utils.SphereCfg(
                radius=0.009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.00, 0.85, 0.05)),
            ),
            "pending": sim_utils.SphereCfg(
                radius=0.009,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.70, 0.70)),
            ),
        },
    )
    """State beads for completed, active, and later ordered route steps."""

    route_direction_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cable_route_directions",
        markers={
            "clockwise": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.040, 0.005, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.00, 0.22, 0.05)),
            ),
            "counterclockwise": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.040, 0.005, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.55, 1.00)),
            ),
            "completed": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.040, 0.005, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.20)),
            ),
        },
    )
    """Tangent arrows: orange is clockwise, blue is counterclockwise, and green is complete."""

    route_arc_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cable_route_target_arcs",
        markers={
            "clockwise": sim_utils.SphereCfg(
                radius=0.003,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.00, 0.22, 0.05)),
            ),
            "counterclockwise": sim_utils.SphereCfg(
                radius=0.003,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.55, 1.00)),
            ),
            "completed": sim_utils.SphereCfg(
                radius=0.003,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.20)),
            ),
        },
    )
    """Ghost cable wraps colored by required direction and completion state."""

    def __post_init__(self) -> None:
        """Validate route-program dimensions and values."""
        if self.max_route_steps < 1:
            raise ValueError("max_route_steps must be positive.")
        if not self.allowed_route_ids:
            raise ValueError("allowed_route_ids must contain at least one route.")
        if len(set(self.allowed_route_ids)) != len(self.allowed_route_ids):
            raise ValueError("allowed_route_ids must not contain duplicates.")
        for route_id in self.allowed_route_ids:
            if route_id < 0 or route_id >= len(self.route_options):
                raise ValueError(f"Route id {route_id} is outside route_options.")
        terminal_signatures: set[frozenset[tuple[int, int]]] = set()
        for route in self.route_options:
            if not route or len(route) > self.max_route_steps:
                raise ValueError("Every route must contain between one and max_route_steps entries.")
            if len({peg_index for peg_index, _ in route}) != len(route):
                raise ValueError("A route cannot repeat a peg because each peg has one aggregate winding state.")
            for peg_index, direction in route:
                if peg_index < 0 or peg_index >= len(self.peg_names):
                    raise ValueError(f"Peg index {peg_index} is outside peg_names.")
                if direction not in (-1, 1):
                    raise ValueError("Route directions must be -1 (CCW) or +1 (CW).")
            terminal_signature = frozenset(route)
            if terminal_signature in terminal_signatures:
                raise ValueError("Route options must have distinct terminal peg-direction goals.")
            terminal_signatures.add(terminal_signature)
        if any(scale <= 0.0 for scale in self.goal_position_scale):
            raise ValueError("goal_position_scale values must be positive.")
        if not math.isfinite(self.radial_cutoff) or self.radial_cutoff <= 0.0:
            raise ValueError("radial_cutoff must be finite and positive.")
        if self.axial_cutoff is not None and (not math.isfinite(self.axial_cutoff) or self.axial_cutoff <= 0.0):
            raise ValueError("axial_cutoff must be None or finite and positive.")
        if self.marker_z_offset < 0.0 or self.marker_step_z_spacing < 0.0:
            raise ValueError("Route marker height offsets must be non-negative.")
        if self.marker_radial_offset <= 0.0:
            raise ValueError("marker_radial_offset must be positive.")
        if self.route_arc_radius <= 0.0 or self.route_arc_samples < 8:
            raise ValueError("route_arc_radius must be positive and route_arc_samples must be at least eight.")
        if self.maximum_completion_winding <= self.completion_winding:
            raise ValueError("maximum_completion_winding must exceed completion_winding.")
        if self.maximum_local_cable_length <= 0.0:
            raise ValueError("maximum_local_cable_length must be positive.")
        settled_bounds = torch.as_tensor(self.settled_cable_bounds_b)
        if settled_bounds.shape != (2, 2) or bool((settled_bounds[:, 0] >= settled_bounds[:, 1]).any()):
            raise ValueError("settled_cable_bounds_b must contain ordered x/y bounds with shape (2, 2).")
        if not self.completion_winding < self.reset_replay.completed_winding < self.maximum_completion_winding:
            raise ValueError(
                "reset_replay.completed_winding must lie between completion_winding and maximum_completion_winding."
            )


class CableRoutingCommand(CommandTerm):
    """Sample a configured route program and track its ordered geometric progress."""

    cfg: CableRoutingCommandCfg

    def __init__(self, cfg: CableRoutingCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.cable = env.scene[cfg.cable_name]
        self.pegs = [env.scene[name] for name in cfg.peg_names]
        control_points = self.cable.cfg.spawn.positions
        edge_count = len(control_points) - 1
        if edge_count < 1:
            raise ValueError("Cable spawn positions must contain at least two control points.")
        self._cable_rest_length_m = (
            math.fsum(math.dist(control_points[index], control_points[index + 1]) for index in range(edge_count))
            / edge_count
        )

        self.route_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.peg_indices = torch.zeros((self.num_envs, cfg.max_route_steps), dtype=torch.long, device=self.device)
        self.directions = torch.zeros((self.num_envs, cfg.max_route_steps), device=self.device)
        self.valid_steps = torch.zeros((self.num_envs, cfg.max_route_steps), dtype=torch.bool, device=self.device)
        self.goal_tokens = torch.zeros((self.num_envs, cfg.max_route_steps, 6), device=self.device)
        self.directed_progress = torch.zeros((self.num_envs, cfg.max_route_steps), device=self.device)
        self.completed_steps = torch.zeros_like(self.valid_steps)
        self.prefix_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.active_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.winding = torch.zeros((self.num_envs, len(cfg.peg_names)), device=self.device)
        self.route_progress_delta = torch.zeros(self.num_envs, device=self.device)
        self._reward_progress_baseline = torch.zeros(self.num_envs, device=self.device)

        self.metrics["route_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["completed_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["active_directed_winding_rad"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_abs_winding_rad"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["route_id"] = torch.zeros(self.num_envs, device=self.device)

        # Like the SO101 typing task, physical replay is owned by the command:
        # reset events run before command reset, so this is the only layer that
        # can restore a cable snapshot and its matching route atomically.
        self._episode_reset = False
        self._last_sampled_sources: torch.Tensor | None = None
        self._reset_metric_last: dict[str, float] = {}
        self._reset_generator = torch.Generator(device=self.device).manual_seed(cfg.reset_replay.seed)
        self.reset_replay: CableResetReplay | None = None
        if cfg.reset_replay.enabled:
            self.reset_replay = CableResetReplay(cfg.reset_replay, env)
        self._arm_joint_ids = {
            name: env.scene[name].find_joints(["joint[1-6]"])[0] for name in ("yam_left", "yam_right")
        }
        self._gripper_joint_ids = {
            name: (
                env.scene[name].find_joints("left_finger")[0][0],
                env.scene[name].find_joints("right_finger")[0][0],
            )
            for name in ("yam_left", "yam_right")
        }
        self._contact_body_ids = {
            name: env.scene[name].find_bodies("link_6")[0][0] for name in ("yam_left", "yam_right")
        }
        self._reset_ik_helpers: dict[str, object] | None = None

    @property
    def command(self) -> torch.Tensor:
        """Return flattened bounded route tokens, shape ``(num_envs, max_route_steps * 6)``."""
        return self.goal_tokens.flatten(start_dim=1)

    @property
    def peg_positions_w(self) -> torch.Tensor:
        """Return current peg centers [m], shape ``(num_envs, num_pegs, 3)``."""
        return torch.stack([peg.data.root_pos_w.torch for peg in self.pegs], dim=1)

    @property
    def active_peg_positions_w(self) -> torch.Tensor:
        """Return the active peg center for every environment [m]."""
        active_peg = torch.gather(self.peg_indices, 1, self.active_step[:, None]).squeeze(1)
        return torch.gather(self.peg_positions_w, 1, active_peg[:, None, None].expand(-1, 1, 3)).squeeze(1)

    def refresh_route_state(self, *, update_reward_delta: bool = False) -> None:
        """Refresh winding and ordered completion from the live scene.

        Args:
            update_reward_delta: Whether to consume the change in normalized ordered progress for
                the current policy step. The termination term sets this once before rewards run.
        """
        cable_points = self.cable.data.segment_pose_w.torch[..., :3]
        peg_positions = self.peg_positions_w
        finite_geometry = torch.isfinite(cable_points).all(dim=(1, 2)) & torch.isfinite(peg_positions).all(dim=(1, 2))
        safe_cable_points = torch.where(finite_geometry[:, None, None], cable_points, torch.zeros_like(cable_points))
        safe_peg_positions = torch.where(finite_geometry[:, None, None], peg_positions, torch.zeros_like(peg_positions))
        self.winding[:] = benchmark_winding_angle(
            safe_cable_points,
            safe_peg_positions,
            self.cfg.radial_cutoff,
            self.cfg.axial_cutoff,
        )
        local_span_count, local_cable_length = benchmark_local_cable_spans(
            safe_cable_points,
            safe_peg_positions,
            self.cfg.radial_cutoff,
            self.cfg.axial_cutoff,
        )
        single_local_span = (local_span_count == 1) & (local_cable_length <= self.cfg.maximum_local_cable_length)
        step_completion_mask = torch.gather(single_local_span, 1, self.peg_indices)
        progress, completed, prefix, success = ordered_route_state(
            self.winding,
            self.peg_indices,
            self.directions,
            self.valid_steps,
            self.cfg.completion_winding,
            maximum_completion_winding=self.cfg.maximum_completion_winding,
            completion_mask=step_completion_mask,
        )
        self.directed_progress[:] = progress
        self.completed_steps[:] = completed
        self.prefix_length[:] = prefix
        self.succeeded |= success & finite_geometry
        route_length = self.valid_steps.sum(dim=1)
        self.active_step[:] = torch.minimum(prefix, route_length - 1)
        if update_reward_delta:
            score = self._route_progress_score()
            finite_score = finite_geometry & torch.isfinite(score) & torch.isfinite(self._reward_progress_baseline)
            self.route_progress_delta[:] = torch.where(
                finite_score,
                score - self._reward_progress_baseline,
                torch.zeros_like(score),
            )
            self._reward_progress_baseline[:] = torch.where(
                finite_score,
                score,
                torch.nan_to_num(self._reward_progress_baseline),
            )
        # Terminations own the geometric refresh before rewards. Keep the
        # logged metrics atomic with that live state so an episode cannot
        # terminate successfully while reporting a stale success metric.
        self._update_metrics()

    def _route_progress_score(self) -> torch.Tensor:
        """Return normalized prefix plus active-step progress."""
        route_length = self.valid_steps.sum(dim=1).clamp(min=1)
        active_progress = torch.gather(self.directed_progress.clamp(min=0.0), 1, self.active_step[:, None]).squeeze(1)
        active_progress = torch.where(self.prefix_length < route_length, active_progress, 0.0)
        return (self.prefix_length + active_progress) / route_length

    def _update_metrics(self) -> None:
        route_length = self.valid_steps.sum(dim=1).clamp(min=1)
        active_winding = torch.gather(self.directed_progress, 1, self.active_step[:, None]).squeeze(1)
        self.metrics["route_progress"][:] = self._route_progress_score()
        self.metrics["success"][:] = self.succeeded.float()
        self.metrics["completed_fraction"][:] = self.prefix_length / route_length
        self.metrics["active_directed_winding_rad"][:] = active_winding
        self.metrics["max_abs_winding_rad"][:] = self.winding.abs().amax(dim=1)
        self.metrics["route_id"][:] = self.route_id.float()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        if self.reset_replay is None or not self._episode_reset:
            # A physical snapshot only owns the route program it was stored
            # with. If the long command timer is ever shortened, a mid-episode
            # goal redraw becomes an ordinary source and must not credit the
            # old snapshot at the next terminal reset.
            if self.reset_replay is not None:
                self.reset_replay.env_source[env_ids] = -1
            self._assign_route_ids(env_ids, self._sample_route_ids(len(env_ids)))
            self.refresh_route_state()
            if self._episode_reset and bool(self.succeeded[env_ids].any()):
                bad = env_ids[self.succeeded[env_ids]].tolist()
                raise RuntimeError(f"Ordinary reset produced terminal-success cable states for environments {bad}.")
            self.route_progress_delta[env_ids] = 0.0
            self._reward_progress_baseline[env_ids] = self._route_progress_score()[env_ids]
            return

        if not self.reset_replay.built:
            all_ids = torch.arange(self.num_envs, device=self.device)
            if len(env_ids) != self.num_envs or not torch.equal(env_ids, all_ids):
                raise RuntimeError(
                    "The cable reset replay bank must be built by the initial full-environment reset. "
                    "Call env.reset() once before requesting partial resets."
                )
            self._build_reset_replay()
        route_ids = self._sample_route_ids(len(env_ids))
        source = self.reset_replay.sample_sources(route_ids)
        self.reset_replay.env_source[env_ids] = source
        self._last_sampled_sources = source

        replayed = source >= 0
        if bool(replayed.any()):
            replay_env_ids = env_ids[replayed]
            replay_rows = source[replayed]
            self.reset_replay.restore(replay_env_ids, replay_rows)
            if not torch.equal(route_ids[replayed], self.reset_replay.route_id[replay_rows]):
                raise RuntimeError("Route-conditioned reset replay returned a row for the wrong goal.")
        self._assign_route_ids(env_ids, route_ids)

        # Derive progress from the restored geometry instead of trusting bank
        # metadata, then seed the ratchet baseline so baked-in near-goal
        # progress earns no free first-step reward.
        self.refresh_route_state()
        if bool(self.succeeded[env_ids].any()):
            bad = env_ids[self.succeeded[env_ids]].tolist()
            raise RuntimeError(f"Reset replay restored terminal-success cable states for environments {bad}.")
        self.route_progress_delta[env_ids] = 0.0
        self._reward_progress_baseline[env_ids] = self._route_progress_score()[env_ids]

    def _sample_route_ids(self, count: int) -> torch.Tensor:
        """Sample allowed route programs uniformly."""
        allowed = torch.tensor(self.cfg.allowed_route_ids, dtype=torch.long, device=self.device)
        sampled = torch.randint(
            len(allowed),
            (count,),
            device=self.device,
            generator=self._reset_generator,
        )
        return allowed[sampled]

    def _assign_route_ids(self, env_ids: torch.Tensor, route_ids: torch.Tensor) -> None:
        """Assign route programs and clear all per-episode command state."""
        if route_ids.shape != env_ids.shape:
            raise ValueError(f"route_ids must have shape {tuple(env_ids.shape)}; got {tuple(route_ids.shape)}.")
        self.route_id[env_ids] = route_ids
        self.peg_indices[env_ids] = 0
        self.directions[env_ids] = 0.0
        self.valid_steps[env_ids] = False
        self.directed_progress[env_ids] = 0.0
        self.completed_steps[env_ids] = False
        self.prefix_length[env_ids] = 0
        self.active_step[env_ids] = 0
        self.succeeded[env_ids] = False
        self.route_progress_delta[env_ids] = 0.0
        self._reward_progress_baseline[env_ids] = 0.0
        self.metrics["route_progress"][env_ids] = 0.0
        self.metrics["success"][env_ids] = 0.0
        self.metrics["completed_fraction"][env_ids] = 0.0
        self.metrics["active_directed_winding_rad"][env_ids] = 0.0
        self.metrics["max_abs_winding_rad"][env_ids] = 0.0
        self.metrics["route_id"][env_ids] = self.route_id[env_ids].float()

        for route_id, route in enumerate(self.cfg.route_options):
            selected_envs = env_ids[self.route_id[env_ids] == route_id]
            if len(selected_envs) == 0:
                continue
            for step, (peg_index, direction) in enumerate(route):
                self.peg_indices[selected_envs, step] = peg_index
                self.directions[selected_envs, step] = float(direction)
                self.valid_steps[selected_envs, step] = True
        self._refresh_goal_tokens(env_ids)

    def _sample_reset_programs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stratify bank rows over routes, active stages, progress, and interaction phase."""
        assert self.reset_replay is not None
        capacity = self.cfg.reset_replay.buffer_size
        slot = torch.arange(capacity, device=self.device)
        allowed = torch.tensor(self.cfg.allowed_route_ids, device=self.device, dtype=torch.long)
        route_ids = allowed[slot % len(allowed)]
        route_lengths = torch.tensor(
            [len(route) for route in self.cfg.route_options],
            device=self.device,
            dtype=torch.long,
        )[route_ids]
        route_cycle = torch.div(slot, len(allowed), rounding_mode="floor")
        active_steps = route_cycle % route_lengths

        # Interleaved deterministic strata make the frozen bank cover the full
        # near-goal interval without imposing route partitions on SuccessMonitor.
        # Each mixed-radix digit is independent so a future three-step route
        # cannot accidentally correlate active step with interaction phase.
        num_progress_bins = 16
        progress_cycle = torch.div(route_cycle, route_lengths, rounding_mode="floor")
        progress_bin = progress_cycle % num_progress_bins
        unit = (progress_bin.to(torch.float32) + 0.5) / num_progress_bins
        progress_min, progress_max = self.cfg.reset_replay.active_progress_range
        progress = progress_min + unit * (progress_max - progress_min)
        active_winding = progress * self.cfg.completion_winding
        interaction_phase = torch.div(progress_cycle, num_progress_bins, rounding_mode="floor") % 3
        return route_ids, active_steps, active_winding, interaction_phase

    def _reset_bank_scratch(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Create safe heterogeneous robot and fixture states for a build batch."""
        env_mdp.reset_scene_to_default(self._env, env_ids, reset_joint_targets=True)
        joint_jitter = self.cfg.reset_replay.arm_joint_position_jitter
        for name, joint_ids in self._arm_joint_ids.items():
            robot = self._env.scene[name]
            position = robot.data.default_joint_pos.torch[env_ids].clone()
            if joint_jitter > 0.0:
                noise = torch.empty(
                    (len(env_ids), len(joint_ids)),
                    device=self.device,
                    dtype=position.dtype,
                ).uniform_(-joint_jitter, joint_jitter, generator=self._reset_generator)
                position[:, joint_ids] += noise
                limits = robot.data.soft_joint_pos_limits.torch[env_ids][:, joint_ids]
                position[:, joint_ids] = position[:, joint_ids].clamp(limits[..., 0], limits[..., 1])
            velocity = torch.zeros_like(position)
            robot.write_joint_position_to_sim_index(position=position, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids)
            robot.set_joint_position_target_index(target=position, env_ids=env_ids)
            robot.set_joint_velocity_target_index(target=velocity, env_ids=env_ids)
        return reset_peg_offsets(
            self._env,
            env_ids,
            asset_names=self.cfg.peg_names,
            generator=self._reset_generator,
        )

    def _ensure_reset_ik_helpers(self) -> dict[str, object]:
        """Lazily construct unregistered Newton IK solvers for bank authoring only."""
        if self._reset_ik_helpers is not None:
            return self._reset_ik_helpers

        from isaaclab_newton.envs.mdp import NewtonInverseKinematicsActionCfg
        from isaaclab_newton.envs.mdp.actions.newton_ik_actions import NewtonInverseKinematicsAction
        from isaaclab_newton.ik import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg, NewtonIKSolverCfg

        self._reset_ik_helpers = {}
        target_cfg = self.cfg.reset_replay.robot_targets
        for name in self._arm_joint_ids:
            helper_cfg = NewtonInverseKinematicsActionCfg(
                asset_name=name,
                joint_names=["joint[1-6]"],
                isolate_articulation_model=True,
                use_cuda_graph=str(self.device).startswith("cuda"),
                controller=NewtonIKSolverCfg(
                    optimizer="lm",
                    jacobian_mode="analytic",
                    sampler="gauss",
                    n_seeds=target_cfg.ik_num_seeds,
                    noise_std=target_cfg.ik_noise_std,
                    iterations=24,
                    lambda_initial=0.05,
                    rng_seed=self.cfg.reset_replay.seed,
                ),
                objectives=[
                    NewtonIKPoseObjectiveCfg(
                        name=f"{name}_reset_contact",
                        body_name="link_6",
                        body_offset_pos=YAM_CONTACT_FRAME_OFFSET_POS,
                        body_offset_rot=YAM_CONTACT_FRAME_OFFSET_QUAT,
                        command_type="pose",
                        use_relative_mode=False,
                        scale=1.0,
                        position_weight=1.0,
                        rotation_weight=2.0,
                    ),
                    NewtonIKJointLimitObjectiveCfg(weight=0.1),
                ],
            )
            self._reset_ik_helpers[name] = NewtonInverseKinematicsAction(helper_cfg, self._env)
        return self._reset_ik_helpers

    def _contact_pose_w(self, robot_name: str, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the physical inner-pad midpoint pose for one YAM."""
        robot = self._env.scene[robot_name]
        body_id = self._contact_body_ids[robot_name]
        body_pos = robot.data.body_pos_w.torch[env_ids, body_id]
        body_quat = robot.data.body_quat_w.torch[env_ids, body_id]
        offset_pos = body_pos.new_tensor(YAM_CONTACT_FRAME_OFFSET_POS).expand(len(env_ids), -1)
        offset_quat = body_quat.new_tensor(YAM_CONTACT_FRAME_OFFSET_QUAT).expand(len(env_ids), -1)
        return combine_frame_transforms(body_pos, body_quat, offset_pos, offset_quat)

    def _solve_reset_ik(
        self,
        robot_name: str,
        env_ids: torch.Tensor,
        target_poses_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve absolute contact targets while preserving targets in unrelated worlds."""
        helpers = self._ensure_reset_ik_helpers()
        helper = helpers[robot_name]
        robot = self._env.scene[robot_name]
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        contact_pos_w, contact_quat_w = self._contact_pose_w(robot_name, all_ids)
        contact_pos_b, contact_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            contact_pos_w,
            contact_quat_w,
        )
        actions = torch.cat((contact_pos_b, contact_quat_b), dim=-1)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w.torch[env_ids],
            robot.data.root_quat_w.torch[env_ids],
            target_poses_w[:, :3],
            target_poses_w[:, 3:7],
        )
        actions[env_ids] = torch.cat((target_pos_b, target_quat_b), dim=-1)

        previous_targets = robot.data.joint_pos_target.torch.clone()
        helper.process_actions(actions)  # type: ignore[attr-defined]
        helper.apply_actions()  # type: ignore[attr-defined]
        arm_joint_ids = self._arm_joint_ids[robot_name]
        solved = robot.data.joint_pos_target.torch[env_ids][:, arm_joint_ids].clone()
        robot.set_joint_position_target_index(target=previous_targets)

        limits = robot.data.soft_joint_pos_limits.torch[env_ids][:, arm_joint_ids]
        valid = torch.isfinite(solved).all(dim=1)
        valid &= (solved >= limits[..., 0]).all(dim=1)
        valid &= (solved <= limits[..., 1]).all(dim=1)
        current = robot.data.joint_pos.torch[env_ids][:, arm_joint_ids]
        solved = torch.where(valid[:, None], solved, current)
        return solved, valid

    def _condition_bank_robots(
        self,
        env_ids: torch.Tensor,
        route_ids: torch.Tensor,
        active_steps: torch.Tensor,
        interaction_phase: torch.Tensor,
    ) -> _RobotResetCondition:
        """Pair generated cables with reachable reach, cage, and bimanual YAM states."""
        target_cfg = self.cfg.reset_replay.robot_targets
        num_rows = len(env_ids)
        assigned = torch.zeros((num_rows, 2), device=self.device, dtype=torch.bool)
        segment_indices = torch.zeros((num_rows, 2), device=self.device, dtype=torch.long)
        height_offsets = torch.zeros((num_rows, 2), device=self.device)
        ik_valid = torch.ones(num_rows, device=self.device, dtype=torch.bool)
        if not target_cfg.enabled or num_rows == 0:
            return _RobotResetCondition(assigned, segment_indices, height_offsets, ik_valid)

        cable_poses = self.cable.data.segment_pose_w.torch[env_ids]
        peg_positions = torch.stack(
            [self._env.scene[name].data.root_pose_w.torch[env_ids, :3] for name in self.cfg.peg_names],
            dim=1,
        )
        active_peg_indices = torch.zeros(num_rows, device=self.device, dtype=torch.long)
        for route_id, route in enumerate(self.cfg.route_options):
            route_rows = (route_ids == route_id).nonzero(as_tuple=False).squeeze(-1)
            for step, (peg_index, _) in enumerate(route):
                step_rows = route_rows[active_steps[route_rows] == step]
                active_peg_indices[step_rows] = peg_index
        row_ids = torch.arange(num_rows, device=self.device)
        robot_names = ("yam_left", "yam_right")
        base_xy = torch.stack(
            [self._env.scene[name].data.root_pos_w.torch[env_ids, :2] for name in robot_names],
            dim=1,
        )
        primary_segment, primary_arm = select_workspace_aware_cable_contact_indices(
            cable_poses,
            peg_positions,
            active_peg_indices,
            base_xy,
            radial_cutoff=target_cfg.radial_cutoff,
            minimum_downstream_offset=target_cfg.downstream_segment_offset,
        )
        assigned[row_ids, primary_arm] = True
        segment_indices[row_ids, primary_arm] = primary_segment

        bimanual = interaction_phase == 2
        if bool(bimanual.any()):
            # Keep the active downstream contact with its nearest arm. Give the
            # other arm the closest *unrouted* material segment that is far
            # enough along the cable to avoid two wrists targeting one spot.
            secondary_arm = 1 - primary_arm
            segment_id = torch.arange(cable_poses.shape[1], device=self.device)
            material_separated = (
                segment_id[None] - primary_segment[:, None]
            ).abs() >= target_cfg.bimanual_segment_separation
            peg_distance = torch.linalg.vector_norm(
                cable_poses[..., None, :2] - peg_positions[:, None, :, :2], dim=-1
            ).amin(dim=-1)
            away_from_pegs = peg_distance > target_cfg.radial_cutoff
            eligible = material_separated & away_from_pegs
            eligible = torch.where(eligible.any(dim=1, keepdim=True), eligible, material_separated)
            secondary_base = base_xy[row_ids, secondary_arm]
            secondary_distance = torch.linalg.vector_norm(cable_poses[..., :2] - secondary_base[:, None], dim=-1)
            secondary_distance = torch.where(eligible, secondary_distance, float("inf"))
            secondary_segment = secondary_distance.argmin(dim=1)
            segment_indices[row_ids[bimanual], secondary_arm[bimanual]] = secondary_segment[bimanual]
            assigned[row_ids[bimanual], secondary_arm[bimanual]] = True

        height_offsets[:] = target_cfg.cage_height
        reach = interaction_phase == 0
        height_offsets[reach] = target_cfg.reach_height
        if bool(bimanual.any()):
            # Bimanual frontier: one arm already cages the downstream strand
            # while the second is in a collision-free pre-grasp above a
            # separated material segment. Two simultaneous teleported cages
            # overconstrain the high-friction cable and overflow contact rows.
            height_offsets[bimanual] = target_cfg.reach_height
            height_offsets[row_ids[bimanual], primary_arm[bimanual]] = target_cfg.cage_height

        for arm_index, robot_name in enumerate(robot_names):
            local_rows = assigned[:, arm_index].nonzero(as_tuple=False).squeeze(-1)
            if len(local_rows) == 0:
                continue
            selected_poses = cable_poses[local_rows, segment_indices[local_rows, arm_index]]
            target_poses_w = build_top_down_yam_contact_target_poses(
                selected_poses,
                base_xy[local_rows, arm_index],
                height_offsets[local_rows, arm_index],
            )
            solved, solved_valid = self._solve_reset_ik(robot_name, env_ids[local_rows], target_poses_w)
            ik_valid[local_rows] &= solved_valid

            robot = self._env.scene[robot_name]
            position = robot.data.joint_pos.torch[env_ids[local_rows]].clone()
            position[:, self._arm_joint_ids[robot_name]] = solved
            left_finger, right_finger = self._gripper_joint_ids[robot_name]
            cage_position = position.new_full((len(local_rows),), target_cfg.cage_gripper_joint_position)
            open_position = robot.data.default_joint_pos.torch[env_ids[local_rows], left_finger]
            arm_reach = height_offsets[local_rows, arm_index] == target_cfg.reach_height
            finger_position = torch.where(arm_reach, open_position, cage_position)
            position[:, left_finger] = finger_position
            position[:, right_finger] = -finger_position
            velocity = torch.zeros_like(position)
            robot.write_joint_position_to_sim_index(position=position, env_ids=env_ids[local_rows])
            robot.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids[local_rows])
            robot.set_joint_position_target_index(target=position, env_ids=env_ids[local_rows])
            robot.set_joint_velocity_target_index(target=velocity, env_ids=env_ids[local_rows])

        return _RobotResetCondition(assigned, segment_indices, height_offsets, ik_valid)

    def _settle_bank_scratch(self) -> None:
        """Advance unobserved scratch worlds while holding their robot joint targets."""
        assert self.reset_replay is not None
        for _ in range(self.reset_replay.cfg.settle_steps):
            if self._env._physics_handles_decimation:
                self._env.scene.write_data_to_sim()
                self._env.sim.step(render=False)
                self._env.scene.update(dt=self._env.step_dt)
            else:
                for _ in range(self._env.cfg.decimation):
                    self._env.scene.write_data_to_sim()
                    self._env.sim.step(render=False)
                    self._env.scene.update(dt=self._env.physics_dt)

    def _post_settle_replay_validity(
        self,
        env_ids: torch.Tensor,
        route_ids: torch.Tensor,
        active_steps: torch.Tensor,
        active_winding: torch.Tensor,
        robot_condition: _RobotResetCondition,
        diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reject dynamically hot, stretched, colliding, or topology-changed states."""
        assert self.reset_replay is not None
        cfg = self.reset_replay.cfg
        poses = self.cable.data.segment_pose_w.torch[env_ids]
        velocities = self.cable.data.segment_velocity_w.torch[env_ids]
        peg_positions = torch.stack(
            [self._env.scene[name].data.root_pose_w.torch[env_ids, :3] for name in self.cfg.peg_names],
            dim=1,
        )
        route_diagnostics: dict[str, torch.Tensor] | None = {} if diagnostics is not None else None
        valid, progress = validate_route_conditioned_cable_poses(
            poses,
            peg_positions,
            self._env.scene.env_origins[env_ids],
            route_ids,
            active_steps,
            active_winding,
            self.cfg.route_options,
            rest_length=self._cable_rest_length(),
            completion_winding=self.cfg.completion_winding,
            maximum_completion_winding=self.cfg.maximum_completion_winding,
            radial_cutoff=self.cfg.radial_cutoff,
            axial_cutoff=self.cfg.axial_cutoff,
            maximum_local_cable_length=self.cfg.maximum_local_cable_length,
            # Replay rows move from their scratch clone to arbitrary environment
            # origins. Preserve a small physical reserve so float32 world-frame
            # relocation cannot turn surface contact into penetration.
            fixture_clearance=cfg.restore_clearance,
            board_bounds_b=self.cfg.settled_cable_bounds_b,
            board_clearance=cfg.restore_clearance,
            requested_progress_tolerance=cfg.post_settle_progress_tolerance,
            maximum_active_progress=cfg.maximum_settled_active_progress,
            diagnostics=route_diagnostics,
        )
        if diagnostics is not None:
            diagnostics["route_geometry"] = valid.clone()
            assert route_diagnostics is not None
            diagnostics.update({f"route_{name}": mask for name, mask in route_diagnostics.items()})
        finite_velocity = torch.isfinite(velocities).all(dim=(1, 2))
        linear_speed = torch.linalg.vector_norm(velocities[..., :3], dim=-1).amax(dim=1)
        angular_speed = torch.linalg.vector_norm(velocities[..., 3:], dim=-1).amax(dim=1)
        valid &= finite_velocity
        valid &= linear_speed <= cfg.max_settle_linear_speed
        valid &= angular_speed <= cfg.max_settle_angular_speed
        if diagnostics is not None:
            diagnostics["finite_velocity"] = finite_velocity
            diagnostics["linear_speed"] = linear_speed <= cfg.max_settle_linear_speed
            diagnostics["angular_speed"] = angular_speed <= cfg.max_settle_angular_speed

        relative_joint_gap = cable_relative_joint_gap(poses, self._cable_rest_length())
        joint_gap_valid = relative_joint_gap.amax(dim=1) <= cfg.max_segment_length_relative_error
        valid &= joint_gap_valid
        if diagnostics is not None:
            diagnostics["joint_gap"] = joint_gap_valid

        target_cfg = cfg.robot_targets
        robot_geometry_valid = torch.ones_like(valid) if diagnostics is not None else None
        if target_cfg.enabled:
            valid &= robot_condition.ik_valid
            if diagnostics is not None:
                diagnostics["robot_ik"] = robot_condition.ik_valid
            local_x = poses.new_tensor((1.0, 0.0, 0.0)).expand(len(env_ids), -1)
            local_z = poses.new_tensor((0.0, 0.0, 1.0)).expand(len(env_ids), -1)
            for arm_index, robot_name in enumerate(("yam_left", "yam_right")):
                assigned_rows = robot_condition.assigned[:, arm_index].nonzero(as_tuple=False).squeeze(-1)
                if len(assigned_rows) == 0:
                    continue
                contact_pos, contact_quat = self._contact_pose_w(robot_name, env_ids[assigned_rows])
                base_xy = self._env.scene[robot_name].data.root_pos_w.torch[env_ids[assigned_rows], :2]
                finite_rows = finite_reset_target_rows(
                    poses[assigned_rows],
                    contact_pos,
                    contact_quat,
                    base_xy,
                )
                valid[assigned_rows] &= finite_rows
                assert robot_geometry_valid is not None or diagnostics is None
                if robot_geometry_valid is not None:
                    robot_geometry_valid[assigned_rows] &= finite_rows
                assigned_rows = assigned_rows[finite_rows]
                if len(assigned_rows) == 0:
                    continue
                contact_pos = contact_pos[finite_rows]
                contact_quat = contact_quat[finite_rows]
                base_xy = base_xy[finite_rows]
                original_segment_indices = robot_condition.segment_indices[assigned_rows, arm_index]
                # A geometric cage permits low-friction tangential sliding. Follow
                # the nearest material point on the same local strand instead of
                # falsely requiring the original segment index to stay fixed.
                cable_query = contact_pos.clone()
                cable_query[:, 2] -= robot_condition.height_offsets[assigned_rows, arm_index]
                live_segment_indices = select_nearest_cable_segment_indices(
                    poses[assigned_rows, :, :3],
                    cable_query,
                    original_segment_indices,
                    search_radius=target_cfg.post_settle_segment_window,
                )
                segment_poses = poses[assigned_rows, live_segment_indices]
                target_geometry_valid = valid_top_down_yam_target_rows(segment_poses)
                valid[assigned_rows] &= target_geometry_valid
                if robot_geometry_valid is not None:
                    robot_geometry_valid[assigned_rows] &= target_geometry_valid
                assigned_rows = assigned_rows[target_geometry_valid]
                if len(assigned_rows) == 0:
                    continue
                contact_pos = contact_pos[target_geometry_valid]
                contact_quat = contact_quat[target_geometry_valid]
                base_xy = base_xy[target_geometry_valid]
                segment_poses = segment_poses[target_geometry_valid]
                live_target = build_top_down_yam_contact_target_poses(
                    segment_poses,
                    base_xy,
                    robot_condition.height_offsets[assigned_rows, arm_index],
                )
                position_error = torch.linalg.vector_norm(contact_pos - live_target[:, :3], dim=-1)
                position_ok = position_error <= target_cfg.max_contact_position_error

                contact_x = quat_apply(contact_quat, local_x[assigned_rows])
                contact_z = quat_apply(contact_quat, local_z[assigned_rows])
                cable_tangent = quat_apply(segment_poses[:, 3:7], local_z[assigned_rows])
                contact_x_xy = torch.nn.functional.normalize(contact_x[:, :2], dim=-1)
                cable_tangent_xy = torch.nn.functional.normalize(cable_tangent[:, :2], dim=-1)
                tangent_alignment = (contact_x_xy * cable_tangent_xy).sum(dim=-1).abs()
                tangent_ok = tangent_alignment >= target_cfg.min_tangent_alignment
                top_down_alignment = -contact_z[:, 2]
                top_down_ok = top_down_alignment >= 0.90
                # A reach state deliberately leaves a vertical air gap, so the
                # relaxed cable can yaw underneath without invalidating the
                # still-correct top-down approach. Tangent agreement is a cage
                # invariant only once the pads constrain the cable.
                is_reach = robot_condition.height_offsets[assigned_rows, arm_index] == target_cfg.reach_height
                arm_valid = position_ok & (tangent_ok | is_reach) & top_down_ok
                valid[assigned_rows] &= arm_valid
                if robot_geometry_valid is not None:
                    robot_geometry_valid[assigned_rows] &= arm_valid
        elif diagnostics is not None:
            diagnostics["robot_ik"] = torch.ones_like(valid)
        if diagnostics is not None:
            assert robot_geometry_valid is not None
            diagnostics["robot_geometry"] = robot_geometry_valid
        complete_state = self._env.scene.get_state(is_relative=True)
        finite_scene = finite_scene_state_rows(complete_state)[env_ids]
        valid &= finite_scene
        if diagnostics is not None:
            diagnostics["finite_scene"] = finite_scene
        return valid, progress

    def _quiesce_bank_scratch(self, env_ids: torch.Tensor) -> None:
        """Store settled positions with zero velocity and matching PD targets."""
        if len(env_ids) == 0:
            return
        cable_velocity = torch.zeros_like(self.cable.data.segment_velocity_w.torch[env_ids])
        self.cable.write_segment_velocity_to_sim_index(segment_velocity=cable_velocity, env_ids=env_ids)
        for name in self._arm_joint_ids:
            robot = self._env.scene[name]
            position = robot.data.joint_pos.torch[env_ids].clone()
            velocity = torch.zeros_like(position)
            robot.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids)
            robot.set_joint_position_target_index(target=position, env_ids=env_ids)
            robot.set_joint_velocity_target_index(target=velocity, env_ids=env_ids)

    def _build_reset_replay(self) -> None:
        """Generate, physically settle, validate, and freeze full-scene reset states."""
        assert self.reset_replay is not None
        replay = self.reset_replay
        build_start_s = time.perf_counter()
        live_state = self._env.scene.get_state(is_relative=True)
        replay.allocate(live_state)
        assert replay.state_buffer is not None
        route_ids, active_steps, active_winding, interaction_phase = self._sample_reset_programs()
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        cable = self.cable
        terminal_failure_context: dict[str, object] | None = None

        try:
            # A settle advances every cloned world, so fill at most one complete
            # world batch at a time. Fixed-sweep curve projection remains bounded
            # by generation_batch_size, then the entire group settles in parallel.
            for bank_start in range(0, replay.cfg.buffer_size, self.num_envs):
                count = min(self.num_envs, replay.cfg.buffer_size - bank_start)
                pending_local = torch.arange(count, device=self.device, dtype=torch.long)
                for attempt in range(replay.cfg.max_settle_attempts):
                    if len(pending_local) == 0:
                        break
                    env_ids = _reset_replay_scratch_env_ids(
                        all_ids,
                        pending_local,
                        attempt,
                        replay.cfg.max_settle_attempts,
                    )
                    # ``bank_rows`` remains the logical destination index.  Only the
                    # scratch clone used to author that destination rotates on retries.
                    bank_rows = bank_start + pending_local
                    peg_offsets = self._reset_bank_scratch(env_ids)

                    for chunk_start in range(0, len(env_ids), replay.cfg.generation_batch_size):
                        chunk_end = min(chunk_start + replay.cfg.generation_batch_size, len(env_ids))
                        chunk_env_ids = env_ids[chunk_start:chunk_end]
                        chunk_rows = bank_rows[chunk_start:chunk_end]
                        peg_positions = torch.stack(
                            [
                                self._env.scene[name].data.root_pose_w.torch[chunk_env_ids, :3]
                                for name in self.cfg.peg_names
                            ],
                            dim=1,
                        )
                        poses, _ = generate_route_conditioned_cable_poses(
                            cable.data.default_segment_pose_w.torch[chunk_env_ids],
                            peg_positions,
                            self._env.scene.env_origins[chunk_env_ids],
                            route_ids[chunk_rows],
                            active_steps[chunk_rows],
                            active_winding[chunk_rows],
                            self.cfg.route_options,
                            rest_length=self._cable_rest_length(),
                            completion_winding=self.cfg.completion_winding,
                            maximum_completion_winding=self.cfg.maximum_completion_winding,
                            completed_winding=replay.cfg.completed_winding,
                            radial_cutoff=self.cfg.radial_cutoff,
                            axial_cutoff=self.cfg.axial_cutoff,
                            maximum_local_cable_length=self.cfg.maximum_local_cable_length,
                            wrap_radius_range=replay.cfg.wrap_radius_range,
                            entry_angle_jitter=replay.cfg.entry_angle_jitter,
                            start_position_jitter=replay.cfg.start_position_jitter,
                            curve_projection_iterations=replay.cfg.curve_projection_iterations,
                            max_rejection_attempts=replay.cfg.max_curve_attempts,
                            generator=self._reset_generator,
                        )
                        velocity = torch.zeros_like(cable.data.default_segment_velocity_w.torch[chunk_env_ids])
                        cable.write_segment_pose_to_sim_index(segment_pose=poses, env_ids=chunk_env_ids)
                        cable.write_segment_velocity_to_sim_index(
                            segment_velocity=velocity,
                            env_ids=chunk_env_ids,
                        )

                    robot_condition = self._condition_bank_robots(
                        env_ids,
                        route_ids[bank_rows],
                        active_steps[bank_rows],
                        interaction_phase[bank_rows],
                    )
                    self._env.scene.write_data_to_sim()
                    self._env.sim.forward()
                    self._env.scene.update(dt=0.0)
                    self._settle_bank_scratch()
                    collect_terminal_diagnostics = attempt + 1 == replay.cfg.max_settle_attempts
                    settled_diagnostics: dict[str, torch.Tensor] | None = {} if collect_terminal_diagnostics else None
                    valid, progress = self._post_settle_replay_validity(
                        env_ids,
                        route_ids[bank_rows],
                        active_steps[bank_rows],
                        active_winding[bank_rows],
                        robot_condition,
                        settled_diagnostics,
                    )
                    settled_progress = progress

                    # The replay acceptance gate has two stages. First validate the
                    # dynamically settled state, including its speed; then quiesce only
                    # preliminary survivors and reproduce the complete scene-state rewrite
                    # plus zero-time ``sim.forward()`` performed by ordinary ``env.reset()``.
                    # Revalidating after that round trip catches constraint-consistency
                    # motion that a same-state forward can miss when Newton invalidates and
                    # rebuilds cable solver state during pose restoration.
                    # Intersecting both predicates prevents zeroed velocities from
                    # rehabilitating a dynamically hot candidate.
                    preliminary_valid = valid.clone()
                    self._quiesce_bank_scratch(env_ids[preliminary_valid])
                    roundtrip_state = self._env.scene.get_state(is_relative=True)
                    self._env.scene.reset(all_ids)
                    self._env.scene.reset_to(roundtrip_state, env_ids=all_ids, is_relative=True)
                    self._env.sim.forward()
                    self._env.scene.update(dt=0.0)
                    forward_diagnostics: dict[str, torch.Tensor] | None = {} if collect_terminal_diagnostics else None
                    forward_valid, forward_progress = self._post_settle_replay_validity(
                        env_ids,
                        route_ids[bank_rows],
                        active_steps[bank_rows],
                        active_winding[bank_rows],
                        robot_condition,
                        forward_diagnostics,
                    )
                    valid &= forward_valid
                    progress = forward_progress
                    if collect_terminal_diagnostics:
                        terminal_failure_context = {
                            "bank_rows": bank_rows.clone(),
                            "env_ids": env_ids.clone(),
                            "peg_offsets": peg_offsets.clone(),
                            "route_ids": route_ids[bank_rows].clone(),
                            "active_steps": active_steps[bank_rows].clone(),
                            "requested_progress": (active_winding[bank_rows] / self.cfg.completion_winding).clone(),
                            "interaction_phase": interaction_phase[bank_rows].clone(),
                            "settled_progress": settled_progress.clone(),
                            "forward_progress": forward_progress.clone(),
                            "settled_diagnostics": settled_diagnostics,
                            "forward_diagnostics": forward_diagnostics,
                        }

                    replay.build_candidate_count += len(env_ids)
                    replay.build_rejection_count += int((~valid).sum())
                    replay.build_max_attempts = max(replay.build_max_attempts, attempt + 1)
                    accepted_env_ids = env_ids[valid]
                    accepted_rows = bank_rows[valid]
                    self._quiesce_bank_scratch(accepted_env_ids)
                    state = self._env.scene.get_state(is_relative=True)
                    replay.state_buffer.store_rows(accepted_rows, state, accepted_env_ids)
                    replay.route_id[accepted_rows] = route_ids[accepted_rows]
                    replay.active_step[accepted_rows] = active_steps[accepted_rows]
                    replay.interaction_phase[accepted_rows] = interaction_phase[accepted_rows]
                    replay.requested_active_progress[accepted_rows] = (
                        active_winding[accepted_rows] / self.cfg.completion_winding
                    )
                    replay.start_progress[accepted_rows] = progress[valid]
                    pending_local = pending_local[~valid]

                if len(pending_local) > 0:
                    failed_rows = (bank_start + pending_local).tolist()
                    failure_details: list[str] = []
                    if terminal_failure_context is not None:
                        context_rows = terminal_failure_context["bank_rows"]
                        assert isinstance(context_rows, torch.Tensor)
                        phase_names = ("reach", "cage", "bimanual")
                        for failed_row in failed_rows:
                            matching = (context_rows == failed_row).nonzero(as_tuple=False).squeeze(-1)
                            if len(matching) != 1:
                                continue
                            index = int(matching[0])
                            settled = terminal_failure_context["settled_diagnostics"]
                            forward = terminal_failure_context["forward_diagnostics"]
                            assert isinstance(settled, dict) and isinstance(forward, dict)
                            settled_failed = [
                                name
                                for name, mask in settled.items()
                                if mask.dtype == torch.bool and not bool(mask[index])
                            ]
                            forward_failed = [
                                name
                                for name, mask in forward.items()
                                if mask.dtype == torch.bool and not bool(mask[index])
                            ]
                            settled_values = {
                                name: float(value[index])
                                for name, value in settled.items()
                                if value.is_floating_point()
                            }
                            forward_values = {
                                name: float(value[index])
                                for name, value in forward.items()
                                if value.is_floating_point()
                            }
                            route_id = int(terminal_failure_context["route_ids"][index])
                            active_step = int(terminal_failure_context["active_steps"][index])
                            requested_progress = float(terminal_failure_context["requested_progress"][index])
                            phase = phase_names[int(terminal_failure_context["interaction_phase"][index])]
                            scratch_env = int(terminal_failure_context["env_ids"][index])
                            peg_offsets_m = terminal_failure_context["peg_offsets"][index].tolist()
                            settled_progress_value = float(terminal_failure_context["settled_progress"][index])
                            forward_progress_value = float(terminal_failure_context["forward_progress"][index])
                            failure_details.append(
                                f"row {failed_row}: route={route_id}, active_step={active_step}, "
                                f"requested_progress={requested_progress:.5f}, phase={phase}, "
                                f"scratch_env={scratch_env}, peg_offsets_m={peg_offsets_m}, "
                                f"settled_progress={settled_progress_value:.5f}, "
                                f"forward_progress={forward_progress_value:.5f}, "
                                f"settled_failed={settled_failed}, forward_failed={forward_failed}, "
                                f"settled_values={settled_values}, forward_values={forward_values}"
                            )
                    detail_suffix = f" Details: {'; '.join(failure_details)}" if failure_details else ""
                    raise RuntimeError(
                        "Unable to generate dynamically settled cable reset states for bank rows "
                        f"{failed_rows} after {replay.cfg.max_settle_attempts} attempts.{detail_suffix}"
                    )
        finally:
            # Building uses the live worlds as vectorized scratch space. Restore
            # the reset event's states even if candidate generation fails.
            self._env.scene.reset_to(live_state, env_ids=all_ids, is_relative=True)

        replay.build_duration_s = time.perf_counter() - build_start_s
        replay.built = True
        route_counts = torch.bincount(replay.route_id, minlength=len(self.cfg.route_options)).tolist()
        phase_counts = torch.bincount(replay.interaction_phase, minlength=3).tolist()
        _LOGGER.info(
            "[cable-routing] reset replay built: "
            f"{replay.cfg.buffer_size} snapshots, route counts={route_counts}, "
            f"phase counts={phase_counts}, progress={float(replay.start_progress.min()):.3f}.."
            f"{float(replay.start_progress.max()):.3f}, "
            f"rejected={replay.build_rejection_count}/{replay.build_candidate_count}, "
            f"max attempts={replay.build_max_attempts}, duration={replay.build_duration_s:.2f}s"
        )

    def _cable_rest_length(self) -> float:
        """Return the cached authored mean cable-control-point edge length [m]."""
        return self._cable_rest_length_m

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Credit finished episodes, then atomically restore state and route."""
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids is None:
            ids = all_ids
        elif isinstance(env_ids, slice):
            ids = all_ids[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        terminal_success = self.succeeded[ids].clone()
        terminal_route = self.route_id[ids].clone()
        source_before_reset: torch.Tensor | None = None
        if self.reset_replay is not None and self.reset_replay.built:
            source_before_reset = self.reset_replay.env_source[ids].clone()
            self.reset_replay.credit(ids, terminal_success)

        split_metrics: dict[str, float] = {}
        if source_before_reset is not None:
            replayed = source_before_reset >= 0
            key = "reset_replay/buffer_success_rate"
            if bool(replayed.any()):
                self._reset_metric_last[key] = float(terminal_success[replayed].float().mean())
            split_metrics[key] = self._reset_metric_last.get(key, 0.0)
            for route_id in self.cfg.allowed_route_ids:
                mask = terminal_route == route_id
                key = f"reset_replay/route_{route_id}_success_rate"
                if bool(mask.any()):
                    self._reset_metric_last[key] = float(terminal_success[mask].float().mean())
                split_metrics[key] = self._reset_metric_last.get(key, 0.0)

        self._episode_reset = True
        try:
            extras = super().reset(ids)
        finally:
            self._episode_reset = False
        extras.update(split_metrics)
        if self.reset_replay is not None and self.reset_replay.built:
            extras.update(self.reset_replay.metrics(self._last_sampled_sources))
        return extras

    def _refresh_goal_tokens(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        positions_b = self.peg_positions_w[env_ids] - self._env.scene.env_origins[env_ids, None, :]
        board_origin = torch.tensor(self.cfg.board_origin_b, device=self.device)
        scale = torch.tensor(self.cfg.goal_position_scale, device=self.device)
        positions_b = ((positions_b - board_origin) / scale).clamp(min=-1.0, max=1.0)
        selected_positions = torch.gather(
            positions_b,
            1,
            self.peg_indices[env_ids, :, None].expand(-1, -1, 3),
        )

        tokens = self.goal_tokens[env_ids]
        valid = self.valid_steps[env_ids]
        tokens.zero_()
        tokens[..., 0] = valid.float()
        tokens[..., 1:4] = torch.where(valid[..., None], selected_positions, 0.0)
        tokens[..., 4] = (valid & (self.directions[env_ids] > 0.0)).float()
        tokens[..., 5] = (valid & (self.directions[env_ids] < 0.0)).float()
        self.goal_tokens[env_ids] = tokens

    def _update_command(self) -> None:
        """Leave route state unchanged after the authoritative termination refresh.

        :class:`ManagerBasedRLEnv` evaluates terminations before rewards, so
        :func:`route_complete` refreshes the live route geometry at the only
        point needed by both consumers. Episode resets also refresh the state
        while assigning the matching route. Recomputing here would duplicate
        the full winding calculation once per policy step.
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "route_step_visualizer"):
                from isaaclab.markers import VisualizationMarkers  # noqa: PLC0415

                self.route_step_visualizer = VisualizationMarkers(self.cfg.route_step_visualizer_cfg)
                self.route_direction_visualizer = VisualizationMarkers(self.cfg.route_direction_visualizer_cfg)
                self.route_arc_visualizer = VisualizationMarkers(self.cfg.route_arc_visualizer_cfg)
            self.route_step_visualizer.set_visibility(True)
            self.route_direction_visualizer.set_visibility(True)
            self.route_arc_visualizer.set_visibility(True)
        elif hasattr(self, "route_step_visualizer"):
            self.route_step_visualizer.set_visibility(False)
            self.route_direction_visualizer.set_visibility(False)
            self.route_arc_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not all(peg.is_initialized for peg in self.pegs):
            return
        marker_data = _route_goal_marker_data(
            self.peg_positions_w,
            self.peg_indices,
            self.directions,
            self.valid_steps,
            self.completed_steps,
            self.active_step,
            (self.cfg.marker_z_offset, self.cfg.marker_step_z_spacing, self.cfg.marker_radial_offset),
            self.cfg.route_arc_radius,
            self.cfg.route_arc_samples,
        )
        self.route_step_visualizer.visualize(
            translations=marker_data.step_positions_w,
            scales=marker_data.step_scales,
            marker_indices=marker_data.step_marker_indices,
        )
        self.route_direction_visualizer.visualize(
            translations=marker_data.direction_positions_w,
            orientations=marker_data.direction_orientations_w,
            scales=marker_data.direction_scales,
            marker_indices=marker_data.direction_marker_indices,
        )
        self.route_arc_visualizer.visualize(
            translations=marker_data.arc_positions_w,
            scales=marker_data.arc_scales,
            marker_indices=marker_data.arc_marker_indices,
        )
