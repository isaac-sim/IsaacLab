# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the cable-routing task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "cable_capsule_clearance_mask",
    "cable_capsule_self_clearance_mask",
    "reset_cable_state",
    "reset_peg_offsets",
    "sample_benchmark_grid_offsets",
    "sample_board_frame_se2",
    "transform_cable_poses_se2",
]


BENCHMARK_GRID_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
"""The eight nonzero fixture offsets used by the ManipulationNet benchmark."""


def _resolve_env_ids(env_ids: torch.Tensor, num_envs: int, device: str | torch.device) -> torch.Tensor:
    """Resolve an index vector or full environment mask to an index vector."""
    env_ids = env_ids.to(device=device)
    if env_ids.dtype == torch.bool:
        if env_ids.ndim != 1 or env_ids.numel() != num_envs:
            raise ValueError(f"Boolean env mask must have shape ({num_envs},); got {tuple(env_ids.shape)}.")
        return env_ids.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.ndim != 1:
        raise ValueError(f"env_ids must be one-dimensional; got shape {tuple(env_ids.shape)}.")
    return env_ids.to(dtype=torch.long)


def sample_benchmark_grid_offsets(
    num_envs: int,
    num_assets: int,
    *,
    grid_pitch: float = 0.01,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample independent benchmark fixture offsets.

    Args:
        num_envs: Number of environment rows to sample.
        num_assets: Number of fixture offsets per environment.
        grid_pitch: Distance represented by one board-grid step [m].
        device: Device of the returned tensor.
        dtype: Floating-point dtype of the returned tensor.
        generator: Optional device-compatible generator for deterministic sampling.

    Returns:
        Sampled planar offsets [m], shape ``(num_envs, num_assets, 2)``.
    """
    if num_envs < 0 or num_assets < 0:
        raise ValueError(f"num_envs and num_assets must be non-negative; got {num_envs} and {num_assets}.")
    if grid_pitch <= 0.0:
        raise ValueError(f"grid_pitch must be positive; got {grid_pitch}.")

    directions = torch.tensor(BENCHMARK_GRID_DIRECTIONS, device=device, dtype=dtype)
    choices = torch.randint(
        len(BENCHMARK_GRID_DIRECTIONS),
        (num_envs, num_assets),
        device=device,
        generator=generator,
    )
    return directions[choices] * grid_pitch


def sample_board_frame_se2(
    num_envs: int,
    *,
    translation_jitter: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        (-0.02, 0.02),
        (-0.02, 0.02),
    ),
    yaw_jitter: tuple[float, float] = (-0.17453292519943295, 0.17453292519943295),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a planar rigid perturbation in the board frame.

    A two-scalar ``translation_jitter`` applies one ``(min, max)`` range to
    both planar axes. Two nested ranges configure the x and y axes separately.

    Args:
        num_envs: Number of transforms to sample.
        translation_jitter: Uniform x/y translation ranges [m].
        yaw_jitter: Uniform yaw range [rad].
        device: Device of the returned tensors.
        dtype: Floating-point dtype of the returned tensors.
        generator: Optional device-compatible generator for deterministic sampling.

    Returns:
        A pair ``(translation_xy, yaw)`` with shapes ``(num_envs, 2)`` and
        ``(num_envs,)``.
    """
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative; got {num_envs}.")

    translation_range = torch.as_tensor(translation_jitter, device=device, dtype=dtype)
    if translation_range.shape == (2,):
        translation_range = translation_range.expand(2, 2)
    elif translation_range.shape != (2, 2):
        raise ValueError(
            "translation_jitter must be one (min, max) range or separate x/y ranges; "
            f"got shape {tuple(translation_range.shape)}."
        )
    yaw_range = torch.as_tensor(yaw_jitter, device=device, dtype=dtype)
    if yaw_range.shape != (2,):
        raise ValueError(f"yaw_jitter must have shape (2,); got {tuple(yaw_range.shape)}.")
    if bool((translation_range[:, 0] > translation_range[:, 1]).any()) or bool(yaw_range[0] > yaw_range[1]):
        raise ValueError("Jitter ranges must be ordered as (min, max).")

    unit_samples = torch.rand((num_envs, 3), device=device, dtype=dtype, generator=generator)
    translation_xy = translation_range[:, 0] + unit_samples[:, :2] * (translation_range[:, 1] - translation_range[:, 0])
    yaw = yaw_range[0] + unit_samples[:, 2] * (yaw_range[1] - yaw_range[0])
    return translation_xy, yaw


def _cable_segment_endpoints_w(segment_poses_w: torch.Tensor, rest_length: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two centerline endpoints of every cable capsule."""
    local_axis = torch.zeros_like(segment_poses_w[..., :3])
    local_axis[..., 2] = 1.0
    half_axis = 0.5 * rest_length * quat_apply(segment_poses_w[..., 3:7], local_axis)
    return segment_poses_w[..., :3] - half_axis, segment_poses_w[..., :3] + half_axis


def cable_capsule_clearance_mask(
    segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    *,
    rest_length: float,
    cable_radius: float = 0.003,
    peg_radius: float = 0.0125,
    fixture_clearance: float = 0.002,
    board_bounds_b: tuple[tuple[float, float], tuple[float, float]] | None = ((-0.15, 0.15), (-0.20, 0.20)),
    board_clearance: float = 0.002,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Check cable-capsule clearance from peg cylinders and board edges.

    Peg clearance is evaluated from each peg center to every capsule centerline
    in the board plane. Board containment evaluates both centerline endpoints
    with the cable radius and requested clearance as an inset margin.

    Args:
        segment_poses_w: Cable segment poses, shape ``(N, S, 7)``.
        peg_positions_w: Peg centers, shape ``(N, P, 3)``. ``P`` may be zero.
        env_origins_w: Board-frame origins in world frame [m], shape ``(N, 3)``.
        rest_length: Cable capsule centerline length [m].
        cable_radius: Cable capsule radius [m].
        peg_radius: Peg cylinder radius [m].
        fixture_clearance: Additional cable-to-peg surface clearance [m].
        board_bounds_b: Board x/y bounds [m] relative to each environment
            origin, or ``None`` to disable the containment check.
        board_clearance: Additional cable-to-board-edge clearance [m].

    Returns:
        Per-environment validity mask, shape ``(N,)``.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    num_envs = segment_poses_w.shape[0]
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[0] != num_envs or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"peg_positions_w must have shape ({num_envs}, P, 3); got {tuple(peg_positions_w.shape)}.")
    if env_origins_w.shape != (num_envs, 3):
        raise ValueError(f"env_origins_w must have shape ({num_envs}, 3); got {tuple(env_origins_w.shape)}.")
    if rest_length <= 0.0:
        raise ValueError(f"rest_length must be positive; got {rest_length}.")
    if min(cable_radius, peg_radius, fixture_clearance, board_clearance) < 0.0:
        raise ValueError("Cable/peg radii and clearances must be non-negative.")

    segment_start_w, segment_end_w = _cable_segment_endpoints_w(segment_poses_w, rest_length)
    clear = torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)
    peg_clear = torch.ones_like(clear)
    board_clear = torch.ones_like(clear)
    if diagnostics is not None:
        infinite_margin = torch.full(
            (num_envs,),
            float("inf"),
            device=segment_poses_w.device,
            dtype=segment_poses_w.dtype,
        )
        peg_margin = infinite_margin
        board_margin = infinite_margin

    if peg_positions_w.shape[1] > 0:
        start_xy = segment_start_w[:, None, :, :2]
        edge_xy = (segment_end_w - segment_start_w)[:, None, :, :2]
        peg_xy = peg_positions_w[:, :, None, :2]
        projection = ((peg_xy - start_xy) * edge_xy).sum(dim=-1)
        projection /= edge_xy.square().sum(dim=-1).clamp_min(torch.finfo(segment_poses_w.dtype).eps)
        closest_xy = start_xy + projection.clamp(0.0, 1.0)[..., None] * edge_xy
        centerline_distance = torch.linalg.vector_norm(peg_xy - closest_xy, dim=-1)
        required_distance = cable_radius + peg_radius + fixture_clearance
        peg_clear = (centerline_distance >= required_distance).all(dim=(1, 2))
        if diagnostics is not None:
            peg_margin = (centerline_distance - required_distance).amin(dim=(1, 2))
        clear &= peg_clear

    if board_bounds_b is not None:
        bounds = torch.as_tensor(board_bounds_b, device=segment_poses_w.device, dtype=segment_poses_w.dtype)
        if bounds.shape != (2, 2) or bool((bounds[:, 0] >= bounds[:, 1]).any()):
            raise ValueError(f"board_bounds_b must contain ordered x/y bounds with shape (2, 2); got {board_bounds_b}.")
        endpoints_b_xy = (
            torch.cat((segment_start_w[..., :2], segment_end_w[..., :2]), dim=1) - env_origins_w[:, None, :2]
        )
        margin = cable_radius + board_clearance
        lower_bound = bounds[None, :, 0] + margin
        upper_bound = bounds[None, :, 1] - margin
        board_clear = (endpoints_b_xy >= lower_bound).all(dim=(1, 2))
        board_clear &= (endpoints_b_xy <= upper_bound).all(dim=(1, 2))
        if diagnostics is not None:
            lower_margin = endpoints_b_xy - lower_bound
            upper_margin = upper_bound - endpoints_b_xy
            board_margin = torch.minimum(lower_margin, upper_margin).amin(dim=(1, 2))
        clear &= board_clear

    if diagnostics is not None:
        diagnostics["peg_clearance"] = peg_clear
        diagnostics["peg_margin_m"] = peg_margin
        diagnostics["board_clearance"] = board_clear
        diagnostics["board_margin_m"] = board_margin

    return clear


def cable_capsule_self_clearance_mask(
    segment_poses_w: torch.Tensor,
    *,
    rest_length: float,
    cable_radius: float = 0.003,
    self_clearance: float = 0.00025,
    neighbor_exclusion: int = 1,
) -> torch.Tensor:
    """Check planar centerline separation between non-neighbor cable capsules.

    Adjacent capsules intentionally share a centerline endpoint and are excluded.
    The calculation is chunked across environments to bound temporary pairwise
    storage during large vectorized resets.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    if rest_length <= 0.0:
        raise ValueError(f"rest_length must be positive; got {rest_length}.")
    if cable_radius < 0.0 or self_clearance < 0.0:
        raise ValueError("Cable radius and self-clearance must be non-negative.")
    if neighbor_exclusion < 0:
        raise ValueError(f"neighbor_exclusion must be non-negative; got {neighbor_exclusion}.")

    num_envs, num_segments = segment_poses_w.shape[:2]
    if num_segments <= neighbor_exclusion + 1:
        return torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)
    segment_start_w, segment_end_w = _cable_segment_endpoints_w(segment_poses_w, rest_length)
    pair_index = torch.arange(num_segments, device=segment_poses_w.device)
    checked_pairs = pair_index[None, :] - pair_index[:, None] > neighbor_exclusion
    required_distance = 2.0 * cable_radius + self_clearance
    clear = torch.ones(num_envs, device=segment_poses_w.device, dtype=torch.bool)

    def point_segment_distance(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        edge = end - start
        projection = ((point - start) * edge).sum(dim=-1)
        projection /= edge.square().sum(dim=-1).clamp_min(torch.finfo(segment_poses_w.dtype).eps)
        closest = start + projection.clamp(0.0, 1.0)[..., None] * edge
        return torch.linalg.vector_norm(point - closest, dim=-1)

    # Roughly two million segment pairs per chunk keeps peak temporaries modest.
    chunk_size = max(1, 2_000_000 // (num_segments * num_segments))
    for chunk_start in range(0, num_envs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_envs)
        start = segment_start_w[chunk_start:chunk_end, :, :2]
        end = segment_end_w[chunk_start:chunk_end, :, :2]
        start_i, end_i = start[:, :, None], end[:, :, None]
        start_j, end_j = start[:, None, :], end[:, None, :]
        distance = torch.minimum(
            torch.minimum(
                point_segment_distance(start_i, start_j, end_j),
                point_segment_distance(end_i, start_j, end_j),
            ),
            torch.minimum(
                point_segment_distance(start_j, start_i, end_i),
                point_segment_distance(end_j, start_i, end_i),
            ),
        )

        # Endpoint distances do not detect a proper 2-D crossing, so mark those
        # pairs explicitly. Collinear overlap is already caught by the distances.
        edge_i = end_i - start_i
        edge_j = end_j - start_j
        relative_start = start_j - start_i

        def cross_2d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]

        denominator = cross_2d(edge_i, edge_j)
        nonparallel = denominator.abs() > 10.0 * torch.finfo(segment_poses_w.dtype).eps
        safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
        first_fraction = cross_2d(relative_start, edge_j) / safe_denominator
        second_fraction = cross_2d(relative_start, edge_i) / safe_denominator
        intersects = (
            nonparallel
            & (first_fraction >= 0.0)
            & (first_fraction <= 1.0)
            & (second_fraction >= 0.0)
            & (second_fraction <= 1.0)
        )
        distance = torch.where(intersects, torch.zeros_like(distance), distance)
        clear[chunk_start:chunk_end] = (distance[:, checked_pairs] >= required_distance).all(dim=1)
    return clear


def transform_cable_poses_se2(
    segment_poses_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    translation_xy_b: torch.Tensor,
    yaw_b: torch.Tensor,
) -> torch.Tensor:
    """Apply one board-frame SE(2) transform to every segment of each cable.

    The board frame is assumed to be axis-aligned with the environment frame
    and to have its origin at :attr:`InteractiveScene.env_origins`. Yaw is
    premultiplied onto each segment orientation.

    Args:
        segment_poses_w: Segment poses ``(x, y, z, qx, qy, qz, qw)`` in world
            frame [m], shape ``(num_envs, num_segments, 7)``.
        env_origins_w: Board-frame origins in world frame [m], shape
            ``(num_envs, 3)``.
        translation_xy_b: Planar translations in board frame [m], shape
            ``(num_envs, 2)``.
        yaw_b: Yaw rotations in board frame [rad], shape ``(num_envs,)``.

    Returns:
        Transformed segment poses with the same shape and dtype as
        ``segment_poses_w``.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(
            f"segment_poses_w must have shape (num_envs, num_segments, 7); got {tuple(segment_poses_w.shape)}."
        )
    num_envs, num_segments = segment_poses_w.shape[:2]
    if env_origins_w.shape != (num_envs, 3):
        raise ValueError(f"env_origins_w must have shape ({num_envs}, 3); got {tuple(env_origins_w.shape)}.")
    if translation_xy_b.shape != (num_envs, 2):
        raise ValueError(f"translation_xy_b must have shape ({num_envs}, 2); got {tuple(translation_xy_b.shape)}.")
    if yaw_b.shape != (num_envs,):
        raise ValueError(f"yaw_b must have shape ({num_envs},); got {tuple(yaw_b.shape)}.")

    transformed = segment_poses_w.clone()
    local_positions = segment_poses_w[..., :3] - env_origins_w[:, None, :]
    cos_yaw = torch.cos(yaw_b)[:, None]
    sin_yaw = torch.sin(yaw_b)[:, None]
    transformed[..., 0] = (
        env_origins_w[:, None, 0]
        + cos_yaw * local_positions[..., 0]
        - sin_yaw * local_positions[..., 1]
        + translation_xy_b[:, None, 0]
    )
    transformed[..., 1] = (
        env_origins_w[:, None, 1]
        + sin_yaw * local_positions[..., 0]
        + cos_yaw * local_positions[..., 1]
        + translation_xy_b[:, None, 1]
    )
    transformed[..., 2] = env_origins_w[:, None, 2] + local_positions[..., 2]

    half_yaw = 0.5 * yaw_b
    yaw_quat = torch.zeros((num_envs, num_segments, 4), device=segment_poses_w.device, dtype=segment_poses_w.dtype)
    yaw_quat[..., 2] = torch.sin(half_yaw)[:, None]
    yaw_quat[..., 3] = torch.cos(half_yaw)[:, None]
    transformed[..., 3:7] = quat_mul(yaw_quat, segment_poses_w[..., 3:7])
    return transformed


def reset_peg_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_names: Sequence[str] = ("peg_0", "peg_1"),
    base_positions_b: Sequence[Sequence[float]] | torch.Tensor | None = None,
    grid_pitch: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reset pegs to base board positions plus nonzero benchmark offsets.

    When ``base_positions_b`` is omitted, each peg's configured default root
    position is used. Explicit base positions may contain x/y or x/y/z. The
    peg orientations and velocities always come from their configured defaults.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices or a full boolean environment mask to reset.
        asset_names: Kinematic rigid-object scene names, one per peg.
        base_positions_b: Base peg positions in board frame [m], shape
            ``(num_pegs, 2 or 3)``. Defaults to the assets' configured poses.
        grid_pitch: Distance represented by one board-grid step [m].
        generator: Optional device-compatible generator for deterministic sampling.

    Returns:
        Applied x/y offsets in board frame [m], shape
        ``(len(env_ids), num_pegs, 2)``.
    """
    env_ids = _resolve_env_ids(env_ids, len(env.scene.env_origins), env.scene.env_origins.device)
    num_reset = int(env_ids.numel())
    num_assets = len(asset_names)
    if num_reset == 0:
        return torch.empty((0, num_assets, 2), device=env.scene.env_origins.device)
    if num_assets == 0:
        raise ValueError("asset_names must contain at least one peg.")

    first_asset = env.scene[asset_names[0]]
    first_default_pose = first_asset.data.default_root_pose.torch
    dtype = first_default_pose.dtype
    device = first_default_pose.device
    env_ids = env_ids.to(device=device)
    env_origins = env.scene.env_origins[env_ids].to(device=device, dtype=dtype)
    offsets = sample_benchmark_grid_offsets(
        num_reset,
        num_assets,
        grid_pitch=grid_pitch,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    bases: torch.Tensor | None = None
    if base_positions_b is not None:
        bases = torch.as_tensor(base_positions_b, device=device, dtype=dtype)
        if bases.ndim != 2 or bases.shape[0] != num_assets or bases.shape[1] not in (2, 3):
            raise ValueError(
                f"base_positions_b must have shape (num_pegs, 2 or 3); got {tuple(bases.shape)} for {num_assets} pegs."
            )

    for asset_index, asset_name in enumerate(asset_names):
        asset = env.scene[asset_name]
        default_pose = asset.data.default_root_pose.torch[env_ids].clone()
        if default_pose.device != device or default_pose.dtype != dtype:
            raise ValueError("All peg assets must use the same device and dtype.")

        local_position = default_pose[:, :3]
        if bases is not None:
            local_position[:, :2] = bases[asset_index, :2]
            if bases.shape[1] == 3:
                local_position[:, 2] = bases[asset_index, 2]
        default_pose[:, :3] = local_position + env_origins
        default_pose[:, :2] += offsets[:, asset_index]
        asset.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=env_ids)

        default_velocity = asset.data.default_root_vel.torch[env_ids].clone()
        asset.write_root_velocity_to_sim_index(root_velocity=default_velocity, env_ids=env_ids)

    return offsets


def reset_cable_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    translation_jitter: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (
        (-0.002, 0.002),
        (-0.002, 0.002),
    ),
    yaw_jitter: tuple[float, float] = (-0.02, 0.02),
    generator: torch.Generator | None = None,
    full_scene_replay_command_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore the authored neutral cable with a small planar perturbation.

    The route command owns full-scene replay during training. In that mode this event
    intentionally leaves the cable untouched because the command restores cable,
    fixtures, and robots atomically after reset events run.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices or a full boolean environment mask to reset.
        asset_cfg: Cable scene entity.
        translation_jitter: Uniform board-frame x/y translation ranges [m].
        yaw_jitter: Uniform board-frame yaw range [rad].
        generator: Optional device-compatible generator for deterministic sampling.
        full_scene_replay_command_name: Command term owning enabled full-scene replay.

    Returns:
        Applied translation [m] and yaw [rad], with shapes ``(N, 2)`` and ``(N,)``.
    """
    if full_scene_replay_command_name is not None:
        command = env.command_manager.get_term(full_scene_replay_command_name)
        replay = getattr(command, "reset_replay", None)
        if replay is not None and replay.cfg.enabled:
            origins = env.scene.env_origins
            selected_ids = _resolve_env_ids(env_ids, len(origins), origins.device)
            return origins.new_zeros((len(selected_ids), 2)), origins.new_zeros(len(selected_ids))

    cable = env.scene[asset_cfg.name]
    default_pose = cable.data.default_segment_pose_w
    default_velocity = cable.data.default_segment_velocity_w
    if default_pose is None or default_velocity is None:
        raise RuntimeError(f"Cable asset '{asset_cfg.name}' has no initialized default segment state.")

    env_ids = _resolve_env_ids(env_ids, len(default_pose.torch), default_pose.torch.device)
    if len(env_ids) == 0:
        return default_pose.torch.new_empty((0, 2)), default_pose.torch.new_empty((0,))

    segment_poses = default_pose.torch[env_ids].clone()
    translation_xy, yaw = sample_board_frame_se2(
        len(env_ids),
        translation_jitter=translation_jitter,
        yaw_jitter=yaw_jitter,
        device=segment_poses.device,
        dtype=segment_poses.dtype,
        generator=generator,
    )
    segment_poses = transform_cable_poses_se2(
        segment_poses,
        env.scene.env_origins[env_ids],
        translation_xy,
        yaw,
    )
    cable.write_segment_pose_to_sim_index(segment_pose=segment_poses, env_ids=env_ids)
    cable.write_segment_velocity_to_sim_index(
        segment_velocity=default_velocity.torch[env_ids],
        env_ids=env_ids,
    )
    return translation_xy, yaw
