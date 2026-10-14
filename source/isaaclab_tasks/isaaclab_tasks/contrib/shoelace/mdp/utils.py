# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared state helpers for the dual-Franka shoelace task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def tail_state(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return three-segment mean tail positions [m] and velocities [m/s]."""
    left_cable = env.scene[cable_cfgs[0].name]
    right_cable = env.scene[cable_cfgs[1].name]
    left_positions = left_cable.data.segment_pose_w.torch[:, :3, :3].mean(dim=1)
    right_positions = right_cable.data.segment_pose_w.torch[:, -3:, :3].mean(dim=1)
    left_velocities = left_cable.data.segment_velocity_w.torch[:, :3, :3].mean(dim=1)
    right_velocities = right_cable.data.segment_velocity_w.torch[:, -3:, :3].mean(dim=1)
    return torch.stack((right_positions, left_positions), dim=1), torch.stack(
        (right_velocities, left_velocities), dim=1
    )


def tail_x_separation(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return the absolute two-tail X-position separation [m]."""
    tail_positions, _ = tail_state(env, cable_cfgs)
    return torch.abs(tail_positions[:, 1, 0] - tail_positions[:, 0, 0])


def tail_outward_x(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
) -> torch.Tensor:
    """Return each tail's outward X offset from the fixed seam midpoint [m], shape [N, 2].

    Columns follow robot-arm order: the left arm pulls the right cable's last three segments
    toward negative X; the right arm pulls the left cable's first three toward positive X.
    Using the seam midpoint makes the offsets invariant to shoe and environment translations.
    """
    left = env.scene[cable_cfgs[0].name].data.segment_pose_w.torch
    right = env.scene[cable_cfgs[1].name].data.segment_pose_w.torch
    center_x = 0.5 * (left[:, -1, 0] + right[:, 0, 0])
    tail_positions, _ = tail_state(env, cable_cfgs)
    return (tail_positions[:, :, 0] - center_x.unsqueeze(1)) * tail_positions.new_tensor([-1.0, 1.0])


def untying_metrics(
    env: ManagerBasedEnv,
    cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
    throat_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure free cable occupancy around the fixed seam midpoint.

    Args:
        env: Shoelace environment.
        cable_cfgs: Left and right cable scene entities, including their fixed seam anchors.
        throat_radius: Radius of the spherical knot-throat region [m].

    Returns:
        Free segment-center counts in the throat, shape [N, 2] in robot-arm order, and finite position/velocity
        flags, shape [N]. Counts alone are not valid for nonfinite cable states. This is a regional
        geometric criterion, not a topological knot classifier.
    """
    cables = [env.scene[cfg.name] for cfg in cable_cfgs]
    left, right = [cable.data.segment_pose_w.torch[..., :3] for cable in cables]
    center = 0.5 * (left[:, -1] + right[:, 0])
    # Exclude the two fixed anchors, as in the original shoelace demo criterion.
    free_positions = torch.cat((left[:, :-1], right[:, 1:]), dim=1)
    distances = torch.linalg.vector_norm(free_positions - center.unsqueeze(1), dim=-1)
    inside = distances < throat_radius
    left_free_count = left.shape[1] - 1
    # Robot-left controls the right cable; report counts in the same order as grasp and pull metrics.
    throat_count = torch.stack((inside[:, left_free_count:].sum(dim=1), inside[:, :left_free_count].sum(dim=1)), dim=1)
    finite = torch.isfinite(left).all(dim=(1, 2)) & torch.isfinite(right).all(dim=(1, 2))
    for cable in cables:
        finite &= torch.isfinite(cable.data.segment_velocity_w.torch).all(dim=(1, 2))
    return throat_count, finite
