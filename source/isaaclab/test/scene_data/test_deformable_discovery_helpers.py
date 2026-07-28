# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for deformable discovery path/count helpers."""

from __future__ import annotations

from isaaclab.scene_data.deformable_discovery import (
    DeformableStageEntry,
    build_deformable_vertex_count_lookup,
    resolve_deformable_vertex_count,
)


def test_build_deformable_vertex_count_lookup_indexes_root_and_sim_mesh():
    entries = [
        DeformableStageEntry(
            root_path="/World/envs/env_0/Deformable",
            sim_mesh_path="/World/envs/env_0/Deformable/geometry/mesh",
            vis_mesh_path="/World/envs/env_0/Deformable/geometry/mesh",
            deformable_type="volume",
            vertex_count=69,
            vis_vertex_count=69,
        )
    ]
    lookup = build_deformable_vertex_count_lookup(entries)
    assert lookup["/World/envs/env_0/Deformable"] == 69
    assert lookup["/World/envs/env_0/Deformable/geometry/mesh"] == 69


def test_resolve_deformable_vertex_count_walks_ancestors():
    lookup = {
        "/World/envs/env_0/Deformable": 69,
        "/World/envs/env_0/Deformable/geometry/mesh": 69,
    }
    assert (
        resolve_deformable_vertex_count(
            "/World/envs/env_0/Deformable/geometry/mesh/points",
            lookup,
            fallback=128,
        )
        == 69
    )
    assert resolve_deformable_vertex_count("/World/other", lookup, fallback=128) == 128


def test_resolve_deformable_vertex_count_matches_env_relative_suffix():
    lookup = {"/World/envs/env_0/Deformable": 30}
    assert (
        resolve_deformable_vertex_count(
            "/World/envs/env_2/Deformable/geometry/mesh",
            lookup,
            fallback=128,
        )
        == 30
    )
