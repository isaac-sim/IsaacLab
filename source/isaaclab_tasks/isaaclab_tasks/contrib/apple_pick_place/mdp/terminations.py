# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apple_on_plate(
    env: ManagerBasedRLEnv,
    radius: float,
    max_height_difference: float,
    min_contact_force: float,
    max_speed: float,
    release_distance: float,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("apple_contact"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the apple rests on the plate and the hands have let go.

    Contact alone is not enough: the apple can touch the plate rim while held or while rolling past.

    Args:
        env: The environment.
        radius: Largest horizontal distance from the apple's root to the plate's root [m].
        max_height_difference: Largest height difference between the apple's and the plate's roots [m].
        min_contact_force: Smallest apple-plate contact force [N].
        max_speed: Largest apple speed [m/s].
        release_distance: Smallest distance from each body in ``hand_cfg`` to the apple's root [m].
        apple_cfg: The apple.
        plate_cfg: The plate.
        contact_cfg: The apple's contact sensor, filtered to the plate.
        hand_cfg: The robot, with the hand bodies that must be clear of the apple.

    Returns:
        Whether each environment succeeded. Shape is ``(num_envs,)``, dtype bool.
    """
    apple = env.scene[apple_cfg.name].data
    plate_pos = env.scene[plate_cfg.name].data.root_pos_w.torch
    apple_pos = apple.root_pos_w.torch
    offset = apple_pos - plate_pos
    on_footprint = (torch.linalg.vector_norm(offset[:, :2], dim=-1) <= radius) & (
        offset[:, 2].abs() <= max_height_difference
    )
    forces = env.scene.sensors[contact_cfg.name].data.force_matrix_w.torch
    in_contact = torch.linalg.vector_norm(forces, dim=-1).flatten(1).amax(dim=1) > min_contact_force
    settled = torch.linalg.vector_norm(apple.root_lin_vel_w.torch, dim=-1) < max_speed
    hands = env.scene[hand_cfg.name].data.body_pos_w.torch[:, hand_cfg.body_ids]
    released = (torch.linalg.vector_norm(hands - apple_pos[:, None], dim=-1) > release_distance).all(dim=1)
    return on_footprint & in_contact & settled & released
