# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terms of the pick-and-place task: the object layout, and the conditions of its success."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab_experimental.mdp2 import State, Term, TermContext
from isaaclab_experimental.mdp2.rng import uniform

# Layout.


def sample_layouts(
    ctx: TermContext,
    pool_size: int,
    object_range: tuple[tuple[float, float], tuple[float, float]],
    target_range: tuple[tuple[float, float], tuple[float, float]],
    min_distance: float,
    seed: int,
    **_,
) -> None:
    """Fill the pool with ``(object x, object y, target x, target y)`` layouts whose object and target are at least
    ``min_distance`` apart, by rejection sampling.

    It runs once, on the host side, so it can call any planner, such as one that checks that the robot reaches the
    object; this one only samples uniformly within the ranges, relative to the environment origin [m].
    """
    pool = ctx.state["pool"]
    generator = torch.Generator(device=pool.device).manual_seed(seed)
    low = torch.tensor([object_range[0][0], object_range[1][0], target_range[0][0], target_range[1][0]])
    high = torch.tensor([object_range[0][1], object_range[1][1], target_range[0][1], target_range[1][1]])
    low, high = low.to(pool.device), high.to(pool.device)
    filled = 0
    while filled < pool_size:
        candidates = low + (high - low) * torch.rand(pool_size, 4, generator=generator, device=pool.device)
        valid = candidates[(candidates[:, :2] - candidates[:, 2:]).norm(dim=1) >= min_distance]
        count = min(len(valid), pool_size - filled)
        pool[filled : filled + count] = valid[:count]
        filled += count


def _reset_layout(
    mask: torch.Tensor,
    rng: torch.Tensor,
    pool: torch.Tensor,
    env_origins: torch.Tensor,
    object_pose: torch.Tensor,
    object_vel: torch.Tensor,
    target_pose: torch.Tensor,
    target_vel: torch.Tensor,
    object_height: float,
    target_height: float,
) -> None:
    index = (uniform(rng, 0.0, float(pool.shape[0])).long()).clamp(max=pool.shape[0] - 1)
    layout = pool[index]
    reset = mask[:, None]
    placements = (
        (object_pose, object_vel, layout[:, :2], object_height),
        (target_pose, target_vel, layout[:, 2:], target_height),
    )
    for pose, vel, xy, height in placements:
        new_pose = torch.zeros_like(pose)
        new_pose[:, :2] = env_origins[:, :2] + xy
        new_pose[:, 2] = env_origins[:, 2] + height
        new_pose[:, 6] = 1.0  # Identity orientation, as (x, y, z, w)
        pose.copy_(torch.where(reset, new_pose, pose))
        vel.copy_(torch.where(reset, torch.zeros_like(vel), vel))


reset_layout = Term(
    torch=_reset_layout,
    state={"pool": State(shape=("pool_size", 4), persistent=True)},
    random=True,
    init=sample_layouts,
)
"""Place the object and the target at a layout drawn from a pool that ``init`` fills once at startup, the pattern for
layouts that are expensive to find. Bind the root poses and velocities of both as outputs."""


# Conditions of success.


@wp.func
def _lifted(world: int, root_link_pose_w: wp.array(dtype=wp.transformf), minimum_height: float) -> float:
    return wp.where(wp.transform_get_translation(root_link_pose_w[world])[2] > minimum_height, 1.0, 0.0)


lifted = Term(
    torch=lambda root_link_pose_w, *, minimum_height: (root_link_pose_w[:, 2] > minimum_height).float(),
    warp=_lifted,
)
"""One where the object is above ``minimum_height`` [m]."""


@wp.func
def _placed(
    world: int,
    object_pose: wp.array(dtype=wp.transformf),
    object_vel: wp.array(dtype=wp.spatial_vectorf),
    target_pose: wp.array(dtype=wp.transformf),
    radius: float,
    maximum_height: float,
    maximum_speed: float,
) -> float:
    offset = wp.transform_get_translation(object_pose[world]) - wp.transform_get_translation(target_pose[world])
    near = wp.length(wp.vec2f(offset[0], offset[1])) < radius and offset[2] > 0.0 and offset[2] < maximum_height
    still = wp.length(wp.spatial_top(object_vel[world])) < maximum_speed
    return wp.where(near and still, 1.0, 0.0)


def _placed_torch(
    object_pose: torch.Tensor,
    object_vel: torch.Tensor,
    target_pose: torch.Tensor,
    radius: float,
    maximum_height: float,
    maximum_speed: float,
) -> torch.Tensor:
    offset = object_pose[:, :3] - target_pose[:, :3]
    near = (offset[:, :2].norm(dim=1) < radius) & (offset[:, 2] > 0.0) & (offset[:, 2] < maximum_height)
    return (near & (object_vel[:, :3].norm(dim=1) < maximum_speed)).float()


placed = Term(torch=_placed_torch, warp=_placed)
"""One where the object rests on the target: within ``radius`` of its center [m], above it by less than
``maximum_height`` [m], and slower than ``maximum_speed`` [m/s]."""


@wp.func
def _moved(
    world: int, root_com_vel_w: wp.array(dtype=wp.spatial_vectorf), moved: wp.array(dtype=float), speed: float
) -> float:
    if wp.length(wp.spatial_top(root_com_vel_w[world])) > speed:
        moved[world] = 1.0
    return moved[world]


def _moved_torch(root_com_vel_w: torch.Tensor, moved: torch.Tensor, speed: float) -> torch.Tensor:
    moved.logical_or_(root_com_vel_w[:, :3].norm(dim=1) > speed)
    return moved.clone()


moved = Term(torch=_moved_torch, warp=_moved, state={"moved": State()})
"""One once the object moved faster than ``speed`` [m/s] in the episode, like Arena's object-moved metric."""
