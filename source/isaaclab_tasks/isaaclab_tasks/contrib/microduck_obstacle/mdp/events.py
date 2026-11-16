# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the task's static obstacle field without overlapping its spawn area."""

import math

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

# Separated cells leave room for the largest wall's bounding circle at any yaw.
# A +/-0.15 m jitter leaves at least 1.1 m between neighbouring cell centres.
OBSTACLE_CELLS = {
    "obstacle": (1.4, 0.0),
    "crate": (1.4, 1.4),
    "tall_box": (0.0, 1.4),
    "low_block": (-1.4, 1.4),
    "cylinder": (-1.4, 0.0),
    "thin_post": (-1.4, -1.4),
    "wall_left": (0.0, -1.4),
    "wall_right": (1.4, -1.4),
}


OBSTACLE_LAYOUT_NAMES = ("ring", "open", "barrier", "slalom", "passage", "thin_low", "blocked", "corner", "scattered")
"""Layout names indexed by the per-environment ``microduck_obstacle_layout`` tensor."""

# Non-ring positions are relative to the reset robot's heading. Unused objects
# remain physical, in separated rear/side cells outside the forward approach.
_PARKED_OBSTACLE_POSES = {
    "obstacle": (-0.8, -1.6, 0.0),
    "crate": (-1.6, -0.8, 0.0),
    "tall_box": (-1.6, 0.0, 0.0),
    "low_block": (-1.6, 0.8, 0.0),
    "cylinder": (-0.8, 1.6, 0.0),
    "thin_post": (0.0, 1.6, 0.0),
    "wall_left": (0.8, 1.6, 0.0),
    "wall_right": (0.8, -1.6, 0.0),
}


def reset_obstacle_field(
    env,
    env_ids,
    jitter: float = 0.15,
    layout: str = "ring",
    blocked_front_distance_range: tuple[float, float] = (0.55, 1.05),
    layout_weights: dict[str, float] | None = None,
    spacing_scale: float = 1.0,
    shuffle_groups: tuple[tuple[str, ...], ...] = (),
):
    """Reset static objects with separated footprints and a clear robot spawn.

    ``ring`` preserves the original origin-relative random cells and arbitrary yaw.
    Other families follow the robot's reset pose, varying approach distances,
    spacing, orientation and obstacle order within separated geometric bounds.
    ``mixed`` samples families uniformly by default; ``balanced`` cycles the available
    families for evaluation. Partial resets update only
    the selected objects and entries of ``env.microduck_obstacle_layout``.

    ``blocked_front_distance_range`` sets the blocked front wall's robot-relative
    center distance before jitter. It does not specify sensor depth or guarantee
    physical spawn clearance for custom bounds.

    ``spacing_scale`` enlarges all layout-center offsets for larger props without
    scaling the objects or the common position jitter. Scattered courses and occupied
    passages use fixed meter bounds for the packaged props instead. Wider pockets need not remain closed.

    ``shuffle_groups`` permutes poses among compatible-size objects on each reset.
    Groups must be disjoint; collision footprints and grounding remain attached to each asset.

    ``layout_weights`` selects the mixed-layout distribution; omitted families have
    zero probability. None samples all families uniformly. Scattered courses put six
    props in two rows ahead, with lateral walls and a clear 2.6 m endpoint. Half
    align both middle props with the initial goal corridor; the rest stagger the rows.
    Half of passage resets use a wider corridor with one shuffled prop inside.
    """
    shuffled_names = [name for group in shuffle_groups for name in group]
    if (
        any(len(group) < 2 for group in shuffle_groups)
        or len(shuffled_names) != len(set(shuffled_names))
        or any(name not in OBSTACLE_CELLS for name in shuffled_names)
    ):
        raise ValueError("Shuffle groups must contain at least two known, disjoint obstacle names.")
    if not math.isfinite(spacing_scale) or spacing_scale < 1.0:
        raise ValueError("Spacing scale must be finite and at least one.")
    if not 0.0 <= jitter <= 0.15:
        raise ValueError("Obstacle cell jitter must be in [0, 0.15] m to preserve layout clearance.")
    if (
        len(blocked_front_distance_range) != 2
        or not all(math.isfinite(value) for value in blocked_front_distance_range)
        or not 0.0 < blocked_front_distance_range[0] <= blocked_front_distance_range[1]
    ):
        raise ValueError("Blocked front distance range must be a finite, positive, ordered pair in meters.")
    if layout_weights is not None and (
        layout != "mixed"
        or not layout_weights
        or any(name not in OBSTACLE_LAYOUT_NAMES for name in layout_weights)
        or any(not math.isfinite(weight) or weight < 0 for weight in layout_weights.values())
        or not math.isfinite(sum(layout_weights.values()))
        or sum(layout_weights.values()) <= 0
    ):
        raise ValueError(
            "Layout weights require mixed layout, known names and finite nonnegative weights with positive sum."
        )
    reset_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        reset_ids = reset_ids[env_ids]
    if layout == "mixed":
        if layout_weights is None:
            layout_ids = torch.randint(len(OBSTACLE_LAYOUT_NAMES), (len(reset_ids),), device=env.device)
        else:
            weights = torch.tensor(
                [layout_weights.get(name, 0.0) for name in OBSTACLE_LAYOUT_NAMES],
                device=env.device,
                dtype=torch.float32,
            )
            layout_ids = (
                torch.multinomial(weights, len(reset_ids), replacement=True) if len(reset_ids) else reset_ids.clone()
            )
    elif layout == "balanced":
        layout_ids = reset_ids % len(OBSTACLE_LAYOUT_NAMES)
    elif layout in OBSTACLE_LAYOUT_NAMES:
        layout_ids = torch.full_like(reset_ids, OBSTACLE_LAYOUT_NAMES.index(layout))
    else:
        raise ValueError(f"Unknown obstacle layout {layout!r}; choose {OBSTACLE_LAYOUT_NAMES}, 'mixed' or 'balanced'.")
    if not hasattr(env, "microduck_obstacle_layout"):
        env.microduck_obstacle_layout = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    env.microduck_obstacle_layout[reset_ids] = layout_ids
    obstacle_names = tuple(OBSTACLE_CELLS)
    front_slot = obstacle_names.index("obstacle")
    if not hasattr(env, "microduck_front_obstacle_index"):
        env.microduck_front_obstacle_index = torch.full(
            (env.num_envs,), front_slot, device=env.device, dtype=torch.long
        )
    env.microduck_front_obstacle_index[reset_ids] = front_slot

    if layout != "ring":
        local_poses = _sample_obstacle_poses(layout_ids, jitter, spacing_scale, blocked_front_distance_range)
        robot = env.scene["robot"]
        robot_xy = robot.data.root_link_pos_w.torch[reset_ids, :2]
        _, _, heading = euler_xyz_from_quat(robot.data.root_link_quat_w.torch[reset_ids])
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        ring = layout_ids == 0

    if shuffle_groups:
        slot_order = torch.arange(len(obstacle_names), device=env.device).expand(len(reset_ids), -1).clone()
        for group in shuffle_groups:
            group_indices = torch.tensor([obstacle_names.index(name) for name in group], device=env.device)
            slot_order[:, group_indices] = group_indices[
                torch.rand((len(reset_ids), len(group)), device=env.device).argsort(dim=-1)
            ]
        ring_centers_xy = torch.tensor(list(OBSTACLE_CELLS.values()), device=env.device)[slot_order] * spacing_scale
        # Track the asset receiving the front pose, rather than its pre-shuffle name.
        env.microduck_front_obstacle_index[reset_ids] = (slot_order == front_slot).long().argmax(dim=-1)
        if layout != "ring":
            # Apply the same slot permutation to x, y and yaw.
            pose_indices = slot_order.unsqueeze(-1).expand(-1, -1, 3)
            local_poses = local_poses.gather(1, pose_indices)

    for object_index, (name, center) in enumerate(OBSTACLE_CELLS.items()):
        asset = env.scene[name]
        root_state = asset.data.default_root_state.torch[reset_ids].clone()
        root_state[:, :3] += env.scene.env_origins[reset_ids]
        ring_center_xy = (
            ring_centers_xy[:, object_index]
            if shuffle_groups
            else torch.tensor(center, device=env.device) * spacing_scale
        )
        if layout == "ring":
            root_state[:, :2] = (
                env.scene.env_origins[reset_ids, :2]
                + ring_center_xy
                + torch.empty_like(root_state[:, :2]).uniform_(-jitter, jitter)
            )
            yaw = torch.empty(len(reset_ids), device=env.device).uniform_(-math.pi, math.pi)
        else:
            local_pose = local_poses[:, object_index]
            root_state[:, 0] = robot_xy[:, 0] + cos_heading * local_pose[:, 0] - sin_heading * local_pose[:, 1]
            root_state[:, 1] = robot_xy[:, 1] + sin_heading * local_pose[:, 0] + cos_heading * local_pose[:, 1]
            yaw = heading + local_pose[:, 2]
            root_state[ring, :2] = (
                env.scene.env_origins[reset_ids[ring], :2]
                + (ring_center_xy[ring] if shuffle_groups else ring_center_xy)
                + torch.empty_like(root_state[ring, :2]).uniform_(-jitter, jitter)
            )
            yaw[ring] = torch.empty_like(yaw[ring]).uniform_(-math.pi, math.pi)
        root_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=reset_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=reset_ids)


def _sample_obstacle_poses(
    layout_ids: torch.Tensor,
    jitter: float,
    spacing_scale: float,
    blocked_front_distance_range: tuple[float, float],
) -> torch.Tensor:
    """Assemble layout recipes in reset-heading coordinates; poses contain x/y/yaw."""
    device = layout_ids.device
    obstacle_names = tuple(OBSTACLE_CELLS)
    front_slot = obstacle_names.index("obstacle")
    parked_poses = torch.tensor([_PARKED_OBSTACLE_POSES[name] for name in obstacle_names], device=device)
    local_poses = parked_poses[None].expand(len(layout_ids), -1, -1).clone()
    local_poses[:, :, :2] += torch.empty_like(local_poses[:, :, :2]).uniform_(-0.08, 0.08)
    local_poses[:, :, 2] = torch.empty_like(local_poses[:, :, 2]).uniform_(-math.pi, math.pi)
    layout_offset = torch.empty((len(layout_ids), 2), device=device).uniform_(-jitter, jitter)
    forward, sideways, yaw = local_poses.unbind(dim=-1)  # Writable (reset, obstacle) views.

    barrier = layout_ids == OBSTACLE_LAYOUT_NAMES.index("barrier")
    forward[barrier, front_slot] = torch.empty_like(layout_offset[barrier, 0]).uniform_(0.65, 1.25)
    sideways[barrier, front_slot] = torch.empty_like(layout_offset[barrier, 0]).uniform_(-0.18, 0.18)
    yaw[barrier, front_slot] = torch.empty_like(layout_offset[barrier, 0]).uniform_(-0.6, 0.6)

    slalom = layout_ids == OBSTACLE_LAYOUT_NAMES.index("slalom")
    sequence = _sample_slalom(int(slalom.sum()), device)
    for slot, name in enumerate(("crate", "tall_box", "cylinder")):
        local_poses[slalom, obstacle_names.index(name)] = sequence[:, slot]

    thin_low = layout_ids == OBSTACLE_LAYOUT_NAMES.index("thin_low")
    layout_offset[thin_low] /= 3.0
    sequence = _sample_two_obstacle_course(int(thin_low.sum()), device)
    for slot, name in enumerate(("thin_post", "low_block")):
        local_poses[thin_low, obstacle_names.index(name)] = sequence[:, slot]

    passage = layout_ids == OBSTACLE_LAYOUT_NAMES.index("passage")
    distance = torch.empty_like(layout_offset[passage, 0]).uniform_(0.95, 1.60)
    half_gap = torch.empty_like(distance).uniform_(0.20, 0.55)
    heading = torch.empty_like(distance).uniform_(-0.35, 0.35)
    _set_passage_walls(local_poses, passage, distance, half_gap, heading)
    _sample_side_wall_approaches(local_poses, layout_ids, layout_offset, blocked_front_distance_range)

    # These earlier layouts scale with prop spacing; the following recipes use physical metres.
    local_poses[:, :, :2] *= spacing_scale
    _sample_occupied_passages(local_poses, passage)
    _sample_scattered_course(local_poses, layout_ids == OBSTACLE_LAYOUT_NAMES.index("scattered"))
    local_poses[:, :, :2] += layout_offset[:, None, :]
    return local_poses


def _sample_slalom(count: int, device: torch.device) -> torch.Tensor:
    """Three consecutive obstacles on alternating sides, mirrored and shuffled."""
    poses = torch.zeros((count, 3, 3), device=device)
    forward, sideways, yaw = poses.unbind(dim=-1)
    forward[:, 0].uniform_(0.60, 0.85)
    spacing = torch.empty_like(forward[:, 1:]).uniform_(0.55, 0.75)
    forward[:, 1:] = forward[:, :1] + spacing.cumsum(-1)
    mirror = torch.randint(0, 2, (count, 1), device=device) * 2 - 1
    sideways[:] = torch.empty_like(sideways).uniform_(0.18, 0.38)
    sideways *= mirror * torch.tensor([1, -1, 1], device=device)
    yaw.uniform_(-0.5, 0.5)
    order = torch.rand((count, 3), device=device).argsort(dim=-1)
    return poses[torch.arange(count, device=device)[:, None], order]


def _sample_two_obstacle_course(count: int, device: torch.device) -> torch.Tensor:
    """Two shuffled encounters with space before the fixed 2.6 m goal."""
    poses = torch.zeros((count, 2, 3), device=device)
    forward, sideways, yaw = poses.unbind(dim=-1)
    forward[:, 0].uniform_(0.55, 0.60)
    forward[:, 1] = forward[:, 0] + torch.empty_like(forward[:, 0]).uniform_(0.50, 0.55)
    sideways.uniform_(-0.12, 0.12)
    yaw.uniform_(-0.15, 0.15)
    order = torch.rand((count, 2), device=device).argsort(dim=-1)
    return poses[torch.arange(count, device=device)[:, None], order]


def _sample_side_wall_approaches(
    local_poses: torch.Tensor,
    layout_ids: torch.Tensor,
    layout_offset: torch.Tensor,
    front_distance_range: tuple[float, float],
) -> None:
    """Place blocked/corner approaches and keep their shared lateral offset zero."""
    device = local_poses.device
    obstacle_names = tuple(OBSTACLE_CELLS)
    front_slot = obstacle_names.index("obstacle")
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    blocked = layout_ids == OBSTACLE_LAYOUT_NAMES.index("blocked")
    corner = layout_ids == OBSTACLE_LAYOUT_NAMES.index("corner")
    selected = blocked | corner
    layout_offset[selected, 1] = 0.0
    forward[selected, front_slot] = torch.empty_like(layout_offset[selected, 0]).uniform_(0.55, 1.05)
    low, high = front_distance_range
    if low != 0.55 or high != 1.05:
        # Reuse the draw: overrides must not change the random stream of other layouts.
        forward[blocked, front_slot] = low + (forward[blocked, front_slot] - 0.55) * ((high - low) / 0.5)
    sideways[selected, front_slot] = 0.0
    yaw[selected, front_slot] = 0.0
    half_gap = torch.empty_like(layout_offset[:, 0]).uniform_(0.21, 0.28)
    half_gap[blocked] = torch.empty_like(half_gap[blocked]).uniform_(0.70, 0.80)
    for wall, mask, side in (("wall_left", selected, 1), ("wall_right", blocked, -1)):
        wall_slot = obstacle_names.index(wall)
        forward[mask, wall_slot] = forward[mask, front_slot] - 0.52
        sideways[mask, wall_slot] = side * (half_gap[mask] + 0.06)
        yaw[mask, wall_slot] = math.pi / 2
    mirror = torch.randint(0, 2, (int(corner.sum()), 1), device=device) * 2 - 1
    sideways[corner] *= mirror
    yaw[corner] *= mirror


def _sample_occupied_passages(local_poses: torch.Tensor, passage: torch.Tensor) -> None:
    """Replace half of clear passages with wide corridors containing one prop."""
    device = local_poses.device
    front_slot = tuple(OBSTACLE_CELLS).index("obstacle")
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    occupied = passage.clone()
    occupied[passage] = torch.rand(int(passage.sum()), device=device) < 0.5
    count = int(occupied.sum())
    distance = torch.empty(count, device=device).uniform_(1.35, 1.50)
    half_width = torch.empty(count, device=device).uniform_(1.0, 1.15)
    heading = torch.empty(count, device=device).uniform_(-0.2, 0.2)
    lateral_offset = torch.empty(count, device=device).uniform_(-0.12, 0.12)
    forward[occupied, front_slot] = distance * heading.cos() - lateral_offset * heading.sin()
    sideways[occupied, front_slot] = distance * heading.sin() + lateral_offset * heading.cos()
    yaw[occupied, front_slot] = heading + torch.empty(count, device=device).uniform_(-0.15, 0.15)
    _set_passage_walls(local_poses, occupied, distance, half_width, heading)


def _set_passage_walls(
    local_poses: torch.Tensor,
    selected: torch.Tensor,
    distance: torch.Tensor,
    half_gap: torch.Tensor,
    heading: torch.Tensor,
) -> None:
    """Place parallel walls around a corridor centerline without drawing randomness."""
    obstacle_names = tuple(OBSTACLE_CELLS)
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    for wall, side in (("wall_left", 1), ("wall_right", -1)):
        wall_slot = obstacle_names.index(wall)
        lateral_offset = side * (half_gap + 0.06)
        forward[selected, wall_slot] = distance * heading.cos() - lateral_offset * heading.sin()
        sideways[selected, wall_slot] = distance * heading.sin() + lateral_offset * heading.cos()
        yaw[selected, wall_slot] = math.pi / 2 + heading


def _sample_scattered_course(local_poses: torch.Tensor, scattered: torch.Tensor) -> None:
    """Place six shuffled prop slots in two open rows and add lateral walls."""
    device = local_poses.device
    obstacle_names = tuple(OBSTACLE_CELLS)
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    # Two rows leave multiple routes instead of enclosing the spawn.
    # Bounds are in meters for the packaged props (largest XY footprint 0.70 x 0.58).
    count = int(scattered.sum())
    row_lateral_offset = torch.empty((count, 2), device=device).uniform_(0.15, 0.35)
    mirror = torch.randint(0, 2, (count,), device=device) * 2 - 1
    row_lateral_offset *= mirror[:, None] * torch.tensor([1.0, -1.0], device=device)
    # Half of courses require two successive encounters along the initial goal corridor.
    # Shift the whole row together so side gaps stay clear for the largest props.
    row_lateral_offset[torch.rand(count, device=device) < 0.5] = 0.0
    for index, name in enumerate(("obstacle", "crate", "tall_box", "low_block", "cylinder", "thin_post")):
        row, column = divmod(index, 3)
        slot = obstacle_names.index(name)
        forward[scattered, slot] = 0.8 + 0.9 * row + torch.empty(count, device=device).uniform_(-0.025, 0.025)
        sideways[scattered, slot] = (
            (column - 1) * 1.15 + row_lateral_offset[:, row] + torch.empty(count, device=device).uniform_(-0.05, 0.05)
        )
        yaw[scattered, slot] = torch.empty(count, device=device).uniform_(-0.15, 0.15)
    for name, side in (("wall_left", 1), ("wall_right", -1)):
        slot = obstacle_names.index(name)
        forward[scattered, slot] = torch.empty(count, device=device).uniform_(1.1, 1.4)
        sideways[scattered, slot] = side * torch.empty(count, device=device).uniform_(2.2, 2.5)
        yaw[scattered, slot] = math.pi / 2
