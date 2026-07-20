# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp locomotion termination terms."""

from types import SimpleNamespace

import numpy as np
import warp as wp
from isaaclab_tasks_experimental.manager_based.locomotion.velocity.mdp.terminations import terrain_out_of_bounds


class _Scene(dict):
    """Minimal scene mapping with terrain configuration."""


def _make_env(terrain_type: str, root_x: float = 0.0):
    root_pos_w = wp.array(np.full((2, 3), (root_x, 0.0, 0.0), dtype=np.float32), dtype=wp.vec3f, device="cpu")
    scene = _Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(warp=root_pos_w))))
    scene.cfg = SimpleNamespace(terrain=SimpleNamespace(terrain_type=terrain_type))
    scene.terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(size=(2.0, 2.0), num_rows=2, num_cols=2, border_width=0.0)
        )
    )
    return SimpleNamespace(scene=scene, num_envs=2, device="cpu")


def test_terrain_out_of_bounds_does_not_share_terrain_state_between_envs():
    """A plane environment must not poison bounds used by a later generated-terrain environment."""
    for attribute in ("_is_warmed_up", "_is_plane", "_half_width", "_half_height"):
        if hasattr(terrain_out_of_bounds, attribute):
            delattr(terrain_out_of_bounds, attribute)

    out = wp.zeros(2, dtype=wp.bool, device="cpu")
    terrain_out_of_bounds(_make_env("plane"), out)
    terrain_out_of_bounds(_make_env("generator", root_x=2.0), out, distance_buffer=0.5)

    np.testing.assert_array_equal(out.numpy(), np.ones(2, dtype=np.bool_))
