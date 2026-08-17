# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure geometry helpers for route-conditioned manipulator reset targets."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_unique

__all__ = [
    "CableResetRobotTargetCfg",
    "build_top_down_contact_target_poses",
    "finite_reset_target_rows",
    "select_nearest_cable_segment_indices",
    "select_workspace_aware_cable_contact_indices",
    "valid_top_down_target_rows",
]


@configclass
class CableResetRobotTargetCfg:
    """Geometry configuration for selecting a cable contact during reset.

    Attributes:
        enabled: Whether route-conditioned robot targets are applied during reset.
        radial_cutoff: Maximum planar distance from the active peg to a local cable segment [m].
        downstream_segment_offset: Smallest number of cable segments to advance beyond the final
            active-peg-local segment before searching the reachable free strand.
        bimanual_segment_separation: Ordered segment separation between the two arm contacts.
        reach_height: World ``+Z`` offset used for the collision-free approach target [m].
        cage_height: World ``+Z`` offset used for the cable-caging target [m].
        cage_gripper_joint_position: Optional embodiment-wide scalar gripper position used to cage
            the cable [m or rad, depending on joint type]. If unset, each manipulator descriptor
            supplies its calibrated value.
        max_contact_position_error: Largest accepted contact-frame position error [m].
        min_tangent_alignment: Smallest accepted absolute tangent-axis cosine similarity.
        post_settle_segment_window: Material-index radius searched after the cable settles.
        ik_num_seeds: Gaussian Newton-IK seeds used for one-time reset-state authoring.
        ik_noise_std: Standard deviation of reset IK seed perturbations [rad].
    """

    enabled: bool = True
    radial_cutoff: float = 0.05
    downstream_segment_offset: int = 1
    bimanual_segment_separation: int = 12
    reach_height: float = 0.055
    cage_height: float = 0.003
    cage_gripper_joint_position: float | None = None
    max_contact_position_error: float = 0.02
    min_tangent_alignment: float = 0.70
    post_settle_segment_window: int = 8
    ik_num_seeds: int = 4
    ik_noise_std: float = 0.35

    def __post_init__(self) -> None:
        """Validate the geometry parameters."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        if not math.isfinite(self.radial_cutoff) or self.radial_cutoff <= 0.0:
            raise ValueError("radial_cutoff must be finite and positive.")
        if isinstance(self.downstream_segment_offset, bool) or not isinstance(self.downstream_segment_offset, int):
            raise TypeError("downstream_segment_offset must be an integer.")
        if self.downstream_segment_offset < 1:
            raise ValueError("downstream_segment_offset must be at least one.")
        if isinstance(self.bimanual_segment_separation, bool) or not isinstance(self.bimanual_segment_separation, int):
            raise TypeError("bimanual_segment_separation must be an integer.")
        if self.bimanual_segment_separation < 1:
            raise ValueError("bimanual_segment_separation must be at least one.")
        for name in (
            "reach_height",
            "cage_height",
            "max_contact_position_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.cage_gripper_joint_position is not None and not math.isfinite(self.cage_gripper_joint_position):
            raise ValueError("cage_gripper_joint_position must be None or finite.")
        if self.max_contact_position_error == 0.0:
            raise ValueError("max_contact_position_error must be positive.")
        if self.reach_height < self.cage_height:
            raise ValueError("reach_height must be greater than or equal to cage_height.")
        if not math.isfinite(self.min_tangent_alignment) or not 0.0 <= self.min_tangent_alignment <= 1.0:
            raise ValueError("min_tangent_alignment must be finite and in [0, 1].")
        if (
            isinstance(self.post_settle_segment_window, bool)
            or not isinstance(self.post_settle_segment_window, int)
            or self.post_settle_segment_window < 0
        ):
            raise ValueError("post_settle_segment_window must be a non-negative integer.")
        if isinstance(self.ik_num_seeds, bool) or not isinstance(self.ik_num_seeds, int) or self.ik_num_seeds < 1:
            raise ValueError("ik_num_seeds must be a positive integer.")
        if not math.isfinite(self.ik_noise_std) or self.ik_noise_std <= 0.0:
            raise ValueError("ik_noise_std must be finite and positive.")


def _validate_finite_floating_tensor(name: str, value: torch.Tensor) -> None:
    """Validate that a tensor is floating point and finite."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")


def finite_reset_target_rows(
    cable_segment_poses_w: torch.Tensor,
    contact_positions_w: torch.Tensor,
    contact_quaternions_w: torch.Tensor,
    robot_base_xy_w: torch.Tensor,
) -> torch.Tensor:
    """Return rows safe to pass into strict reset-target geometry helpers.

    Newton may report a non-finite body pose after an over-constrained trial.
    Reset-bank construction is deliberately rejection based, so those trials
    should be regenerated instead of reaching a strict geometry helper and
    terminating the whole training job.
    """
    values = {
        "cable_segment_poses_w": cable_segment_poses_w,
        "contact_positions_w": contact_positions_w,
        "contact_quaternions_w": contact_quaternions_w,
        "robot_base_xy_w": robot_base_xy_w,
    }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype.")

    num_rows = cable_segment_poses_w.shape[0]
    if cable_segment_poses_w.ndim != 3 or cable_segment_poses_w.shape[-1] != 7:
        raise ValueError(f"cable_segment_poses_w must have shape (N, S, 7); got {tuple(cable_segment_poses_w.shape)}.")
    if contact_positions_w.shape != (num_rows, 3):
        raise ValueError(
            f"contact_positions_w must have shape ({num_rows}, 3); got {tuple(contact_positions_w.shape)}."
        )
    if contact_quaternions_w.shape != (num_rows, 4):
        raise ValueError(
            f"contact_quaternions_w must have shape ({num_rows}, 4); got {tuple(contact_quaternions_w.shape)}."
        )
    if robot_base_xy_w.shape != (num_rows, 2):
        raise ValueError(f"robot_base_xy_w must have shape ({num_rows}, 2); got {tuple(robot_base_xy_w.shape)}.")
    if len({value.device for value in values.values()}) != 1:
        raise ValueError("Cable, contact, and robot-base tensors must be on the same device.")

    return (
        torch.isfinite(cable_segment_poses_w).all(dim=(1, 2))
        & torch.isfinite(contact_positions_w).all(dim=1)
        & torch.isfinite(contact_quaternions_w).all(dim=1)
        & torch.isfinite(robot_base_xy_w).all(dim=1)
    )


def valid_top_down_target_rows(cable_segment_poses_w: torch.Tensor) -> torch.Tensor:
    """Return rows whose cable frame can define a planar top-down target."""
    if not isinstance(cable_segment_poses_w, torch.Tensor):
        raise TypeError("cable_segment_poses_w must be a torch.Tensor.")
    if not cable_segment_poses_w.is_floating_point():
        raise TypeError("cable_segment_poses_w must have a floating-point dtype.")
    if cable_segment_poses_w.ndim != 2 or cable_segment_poses_w.shape[-1] != 7:
        raise ValueError(f"cable_segment_poses_w must have shape (N, 7); got {tuple(cable_segment_poses_w.shape)}.")

    finite = torch.isfinite(cable_segment_poses_w).all(dim=1)
    quaternion = cable_segment_poses_w[:, 3:7]
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
    eps = torch.finfo(cable_segment_poses_w.dtype).eps
    normalized_quaternion = quaternion / quaternion_norm.clamp_min(eps)[:, None]
    local_tangent = torch.zeros_like(cable_segment_poses_w[:, :3])
    local_tangent[:, 2] = 1.0
    tangent_xy = quat_apply(normalized_quaternion, local_tangent)[:, :2]
    tangent_norm = torch.linalg.vector_norm(tangent_xy, dim=-1)
    return finite & (quaternion_norm > eps) & (tangent_norm > 32.0 * eps)


def select_workspace_aware_cable_contact_indices(
    cable_segment_poses_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    active_peg_indices: torch.Tensor,
    robot_base_xy_w: torch.Tensor,
    *,
    radial_cutoff: float,
    minimum_downstream_offset: int = 1,
    maximum_planar_reach: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly select a downstream cable segment and the closest robot arm.

    Candidate material segments begin after the final segment local to the
    active peg. Segments local to any peg are excluded whenever at least one
    downstream, in-reach free segment exists. The remaining segment/arm pairs
    are ranked by planar distance to the corresponding robot base. Exact ties
    deterministically prefer the lower segment index and then the lower arm
    index.

    This is a geometric workspace filter, not an IK validity test. Callers must
    retain their strict IK solve and post-solve pose checks for the returned
    contact.

    Args:
        cable_segment_poses_w: Ordered cable-segment poses
            ``(x, y, z, qx, qy, qz, qw)`` in world frame [m, quaternion],
            shape ``(N, S, 7)``.
        peg_positions_w: Peg centers in world frame [m], shape ``(N, P, 3)``.
        active_peg_indices: Active peg index for each row, shape ``(N,)``.
        robot_base_xy_w: Planar robot-base positions in world frame [m],
            shape ``(N, A, 2)``.
        radial_cutoff: Maximum planar distance from a peg to a local segment [m].
        minimum_downstream_offset: Smallest ordered material-index offset after
            the final active-peg-local segment.
        maximum_planar_reach: Optional hard planar base-to-contact distance [m].
            Rows without any downstream pair inside this bound are rejected.

    Returns:
        A pair containing selected cable-segment indices and robot-arm indices,
        each with shape ``(N,)``.

    Raises:
        ValueError: If inputs are invalid, an active peg has no local segment,
            no segment exists at the requested downstream offset, or no
            segment/arm pair lies within :paramref:`maximum_planar_reach`.
    """
    _validate_finite_floating_tensor("cable_segment_poses_w", cable_segment_poses_w)
    _validate_finite_floating_tensor("peg_positions_w", peg_positions_w)
    _validate_finite_floating_tensor("robot_base_xy_w", robot_base_xy_w)
    if cable_segment_poses_w.ndim != 3 or cable_segment_poses_w.shape[-1] != 7:
        raise ValueError(f"cable_segment_poses_w must have shape (N, S, 7); got {tuple(cable_segment_poses_w.shape)}.")
    num_rows, num_segments = cable_segment_poses_w.shape[:2]
    if num_segments < 1:
        raise ValueError("cable_segment_poses_w must contain at least one cable segment.")
    if peg_positions_w.ndim != 3 or peg_positions_w.shape[-1] != 3:
        raise ValueError(f"peg_positions_w must have shape (N, P, 3); got {tuple(peg_positions_w.shape)}.")
    if peg_positions_w.shape[0] != num_rows or peg_positions_w.shape[1] < 1:
        raise ValueError("Cable poses and non-empty peg positions must contain the same number of rows.")
    if robot_base_xy_w.ndim != 3 or robot_base_xy_w.shape[0] != num_rows or robot_base_xy_w.shape[2] != 2:
        raise ValueError(f"robot_base_xy_w must have shape ({num_rows}, A, 2); got {tuple(robot_base_xy_w.shape)}.")
    if robot_base_xy_w.shape[1] < 1:
        raise ValueError("robot_base_xy_w must contain at least one robot arm.")
    if not isinstance(active_peg_indices, torch.Tensor):
        raise TypeError("active_peg_indices must be a torch.Tensor.")
    if active_peg_indices.shape != (num_rows,):
        raise ValueError(f"active_peg_indices must have shape ({num_rows},); got {tuple(active_peg_indices.shape)}.")
    if active_peg_indices.dtype == torch.bool or active_peg_indices.is_floating_point():
        raise TypeError("active_peg_indices must have an integer dtype.")
    input_devices = {
        cable_segment_poses_w.device,
        peg_positions_w.device,
        active_peg_indices.device,
        robot_base_xy_w.device,
    }
    if len(input_devices) != 1:
        raise ValueError("Cable poses, peg positions, active peg indices, and robot bases must be on the same device.")
    if not math.isfinite(radial_cutoff) or radial_cutoff <= 0.0:
        raise ValueError("radial_cutoff must be finite and positive.")
    if (
        isinstance(minimum_downstream_offset, bool)
        or not isinstance(minimum_downstream_offset, int)
        or minimum_downstream_offset < 1
    ):
        raise ValueError("minimum_downstream_offset must be a positive integer.")
    if maximum_planar_reach is not None:
        if (
            isinstance(maximum_planar_reach, bool)
            or not math.isfinite(maximum_planar_reach)
            or maximum_planar_reach <= 0.0
        ):
            raise ValueError("maximum_planar_reach must be None or finite and positive.")

    active_peg_indices = active_peg_indices.to(dtype=torch.long)
    if bool(((active_peg_indices < 0) | (active_peg_indices >= peg_positions_w.shape[1])).any()):
        raise ValueError("active_peg_indices contains an out-of-range peg index.")
    if num_rows == 0:
        empty = active_peg_indices.clone()
        return empty, empty.clone()

    rows = torch.arange(num_rows, device=cable_segment_poses_w.device)
    segment_indices = torch.arange(num_segments, device=cable_segment_poses_w.device)
    segment_xy = cable_segment_poses_w[..., :2]
    active_peg_xy = peg_positions_w[rows, active_peg_indices, :2]
    active_peg_distance = torch.linalg.vector_norm(segment_xy - active_peg_xy[:, None], dim=-1)
    active_local = active_peg_distance <= radial_cutoff
    last_active_local = torch.where(active_local, segment_indices[None], -1).amax(dim=1)
    if bool((last_active_local < 0).any()):
        invalid_rows = (last_active_local < 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        raise ValueError(f"No cable segment lies within radial_cutoff of the active peg for rows {invalid_rows}.")

    first_downstream = last_active_local + minimum_downstream_offset
    if bool((first_downstream >= num_segments).any()):
        invalid_rows = (first_downstream >= num_segments).nonzero(as_tuple=False).squeeze(-1).tolist()
        raise ValueError(f"No cable segment exists at the requested downstream offset for rows {invalid_rows}.")
    downstream = segment_indices[None] >= first_downstream[:, None]

    distance_to_any_peg = torch.linalg.vector_norm(
        segment_xy[:, :, None] - peg_positions_w[:, None, :, :2], dim=-1
    ).amin(dim=-1)
    away_from_pegs = distance_to_any_peg > radial_cutoff
    base_distance = torch.linalg.vector_norm(segment_xy[:, :, None] - robot_base_xy_w[:, None], dim=-1)
    if maximum_planar_reach is not None:
        joint_eligible = downstream[:, :, None] & (base_distance <= maximum_planar_reach)
        has_reachable_pair = joint_eligible.flatten(start_dim=1).any(dim=1)
        if bool((~has_reachable_pair).any()):
            invalid_rows = (~has_reachable_pair).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise ValueError(
                f"No downstream segment/arm pair lies within maximum_planar_reach for rows {invalid_rows}."
            )
    else:
        joint_eligible = downstream[:, :, None].expand_as(base_distance)

    free_joint_eligible = joint_eligible & away_from_pegs[:, :, None]
    has_free_pair = free_joint_eligible.flatten(start_dim=1).any(dim=1)
    eligible = torch.where(has_free_pair[:, None, None], free_joint_eligible, joint_eligible)
    score = torch.where(eligible, base_distance, float("inf"))
    flat_selection = score.flatten(start_dim=1).argmin(dim=1)
    num_arms = robot_base_xy_w.shape[1]
    selected_segment = torch.div(flat_selection, num_arms, rounding_mode="floor")
    selected_arm = flat_selection.remainder(num_arms)
    return selected_segment, selected_arm


def select_nearest_cable_segment_indices(
    cable_segment_positions_w: torch.Tensor,
    query_positions_w: torch.Tensor,
    center_segment_indices: torch.Tensor,
    *,
    search_radius: int,
) -> torch.Tensor:
    """Track the nearest material segment inside a bounded index neighborhood.

    Caging constrains the cable geometrically but intentionally permits sliding
    along its tangent. Searching a local material window therefore validates the
    same physical strand without incorrectly requiring the originally targeted
    segment index to remain between the pads.

    Args:
        cable_segment_positions_w: Cable-segment centers, shape ``(N, S, 3)``.
        query_positions_w: Query position for each row, shape ``(N, 3)``.
        center_segment_indices: Original material segment for each row, shape ``(N,)``.
        search_radius: Number of material indices searched on either side.

    Returns:
        Nearest segment index for every row, shape ``(N,)``.
    """
    _validate_finite_floating_tensor("cable_segment_positions_w", cable_segment_positions_w)
    _validate_finite_floating_tensor("query_positions_w", query_positions_w)
    if cable_segment_positions_w.ndim != 3 or cable_segment_positions_w.shape[-1] != 3:
        raise ValueError(
            f"cable_segment_positions_w must have shape (N, S, 3); got {tuple(cable_segment_positions_w.shape)}."
        )
    if cable_segment_positions_w.shape[1] < 1:
        raise ValueError("cable_segment_positions_w must contain at least one segment.")
    if query_positions_w.shape != (cable_segment_positions_w.shape[0], 3):
        raise ValueError(
            f"query_positions_w must have shape ({cable_segment_positions_w.shape[0]}, 3); "
            f"got {tuple(query_positions_w.shape)}."
        )
    if not isinstance(center_segment_indices, torch.Tensor):
        raise TypeError("center_segment_indices must be a torch.Tensor.")
    if center_segment_indices.shape != (cable_segment_positions_w.shape[0],):
        raise ValueError(
            f"center_segment_indices must have shape ({cable_segment_positions_w.shape[0]},); "
            f"got {tuple(center_segment_indices.shape)}."
        )
    if center_segment_indices.dtype == torch.bool or center_segment_indices.is_floating_point():
        raise TypeError("center_segment_indices must have an integer dtype.")
    input_devices = {
        cable_segment_positions_w.device,
        query_positions_w.device,
        center_segment_indices.device,
    }
    if len(input_devices) != 1:
        raise ValueError("Cable positions, query positions, and center indices must be on the same device.")
    if isinstance(search_radius, bool) or not isinstance(search_radius, int) or search_radius < 0:
        raise ValueError("search_radius must be a non-negative integer.")

    num_segments = cable_segment_positions_w.shape[1]
    center_segment_indices = center_segment_indices.to(dtype=torch.long)
    if bool(((center_segment_indices < 0) | (center_segment_indices >= num_segments)).any()):
        raise ValueError("center_segment_indices contains an out-of-range segment index.")

    offsets = torch.arange(
        -search_radius,
        search_radius + 1,
        device=cable_segment_positions_w.device,
        dtype=torch.long,
    )
    candidates = (center_segment_indices[:, None] + offsets[None]).clamp(0, num_segments - 1)
    rows = torch.arange(len(cable_segment_positions_w), device=cable_segment_positions_w.device)[:, None]
    candidate_positions = cable_segment_positions_w[rows, candidates]
    distances = torch.linalg.vector_norm(candidate_positions - query_positions_w[:, None], dim=-1)
    return torch.gather(candidates, 1, distances.argmin(dim=1, keepdim=True)).squeeze(1)


def build_top_down_contact_target_poses(
    cable_segment_poses_w: torch.Tensor,
    robot_base_xy_w: torch.Tensor,
    height_offsets: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Build top-down manipulator contact targets from selected cable-segment poses.

    The target's local ``+X`` axis follows the cable tangent, with its sign
    selected to point toward the corresponding robot base. Local ``+Z`` points
    down in world frame, and local ``+Y`` is ``(t_y, -t_x, 0)``. Cable segment
    poses use local ``+Z`` as their material tangent axis.

    Args:
        cable_segment_poses_w: Selected cable-segment poses
            ``(x, y, z, qx, qy, qz, qw)`` in world frame [m, quaternion], shape ``(N, 7)``.
        robot_base_xy_w: Planar robot-base positions in world frame [m], shape ``(N, 2)``.
        height_offsets: Signed world ``+Z`` offsets from the cable centers [m],
            either a scalar or one value per row with shape ``(N,)``.

    Returns:
        Target poses ``(x, y, z, qx, qy, qz, qw)`` in world frame, shape ``(N, 7)``.

    Raises:
        ValueError: If shapes or values are invalid, a cable tangent has no
            planar component, or the constructed target is non-finite.
    """
    _validate_finite_floating_tensor("cable_segment_poses_w", cable_segment_poses_w)
    _validate_finite_floating_tensor("robot_base_xy_w", robot_base_xy_w)
    if cable_segment_poses_w.ndim != 2 or cable_segment_poses_w.shape[-1] != 7:
        raise ValueError(f"cable_segment_poses_w must have shape (N, 7); got {tuple(cable_segment_poses_w.shape)}.")
    if robot_base_xy_w.shape != (cable_segment_poses_w.shape[0], 2):
        raise ValueError(
            f"robot_base_xy_w must have shape ({cable_segment_poses_w.shape[0]}, 2); "
            f"got {tuple(robot_base_xy_w.shape)}."
        )
    if cable_segment_poses_w.device != robot_base_xy_w.device:
        raise ValueError("Cable poses and robot base positions must be on the same device.")

    if isinstance(height_offsets, torch.Tensor):
        _validate_finite_floating_tensor("height_offsets", height_offsets)
        if height_offsets.device != cable_segment_poses_w.device:
            raise ValueError("height_offsets and cable poses must be on the same device.")
        if height_offsets.ndim == 0:
            height_offsets = height_offsets.expand(cable_segment_poses_w.shape[0])
        elif height_offsets.shape != (cable_segment_poses_w.shape[0],):
            raise ValueError(
                f"height_offsets must be scalar or have shape ({cable_segment_poses_w.shape[0]},); "
                f"got {tuple(height_offsets.shape)}."
            )
        height_offsets = height_offsets.to(dtype=cable_segment_poses_w.dtype)
    else:
        height_offset = float(height_offsets)
        if not math.isfinite(height_offset):
            raise ValueError("height_offsets must be finite.")
        height_offsets = cable_segment_poses_w.new_full((len(cable_segment_poses_w),), height_offset)

    quaternion_norm = torch.linalg.vector_norm(cable_segment_poses_w[:, 3:7], dim=-1)
    if bool((quaternion_norm <= torch.finfo(cable_segment_poses_w.dtype).eps).any()):
        raise ValueError("Cable segment quaternions must have non-zero norm.")
    cable_quaternion = cable_segment_poses_w[:, 3:7] / quaternion_norm[:, None]
    local_tangent = torch.zeros_like(cable_segment_poses_w[:, :3])
    local_tangent[:, 2] = 1.0
    tangent_xy = quat_apply(cable_quaternion, local_tangent)[:, :2]
    tangent_norm = torch.linalg.vector_norm(tangent_xy, dim=-1)
    if bool((tangent_norm <= 32.0 * torch.finfo(cable_segment_poses_w.dtype).eps).any()):
        raise ValueError("Cable segment tangents must have a non-zero planar component.")
    tangent_xy = tangent_xy / tangent_norm[:, None]

    toward_base_xy = robot_base_xy_w - cable_segment_poses_w[:, :2]
    points_toward_base = (tangent_xy * toward_base_xy).sum(dim=-1) >= 0.0
    tangent_xy = torch.where(points_toward_base[:, None], tangent_xy, -tangent_xy)

    target_x = torch.nn.functional.pad(tangent_xy, (0, 1))
    target_y = torch.stack((tangent_xy[:, 1], -tangent_xy[:, 0], torch.zeros_like(tangent_xy[:, 0])), dim=-1)
    target_z = torch.zeros_like(target_x)
    target_z[:, 2] = -1.0
    target_rotation = torch.stack((target_x, target_y, target_z), dim=-1)
    target_quaternion = quat_unique(quat_from_matrix(target_rotation))

    target_position = cable_segment_poses_w[:, :3].clone()
    target_position[:, 2] += height_offsets
    target_poses_w = torch.cat((target_position, target_quaternion), dim=-1)
    if not bool(torch.isfinite(target_poses_w).all()):
        raise ValueError("Constructed contact target poses must be finite.")
    return target_poses_w
