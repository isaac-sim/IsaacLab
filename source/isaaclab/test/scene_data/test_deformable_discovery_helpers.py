# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for deformable discovery path/count helpers."""

from __future__ import annotations

from isaaclab.scene_data.deformable_discovery import (
    DeformableStageEntry,
    build_deformable_root_path_lookup,
    build_deformable_vertex_count_lookup,
    group_deformable_root_paths_for_views,
    resolve_deformable_root_path,
    resolve_deformable_vertex_count,
    sort_deformable_entries_for_geometry_sync,
)


def test_deformable_vertex_count_lookup_and_resolve():
    """Vertex-count lookup indexes root/sim/vis paths and resolves ancestors and env clones."""
    entries = [
        DeformableStageEntry(
            root_path="/World/envs/env_0/Deformable",
            sim_mesh_path="/World/envs/env_0/Deformable/geometry/mesh",
            vis_mesh_path="/World/envs/env_0/Deformable/geometry/vis_mesh",
            deformable_type="volume",
            vertex_count=69,
            vis_vertex_count=72,
        )
    ]
    lookup = build_deformable_vertex_count_lookup(entries)
    assert lookup["/World/envs/env_0/Deformable"] == 69
    assert lookup["/World/envs/env_0/Deformable/geometry/mesh"] == 69
    assert lookup["/World/envs/env_0/Deformable/geometry/vis_mesh"] == 72
    assert (
        resolve_deformable_vertex_count(
            "/World/envs/env_0/Deformable/geometry/mesh/points",
            lookup,
            fallback=128,
        )
        == 69
    )
    assert (
        resolve_deformable_vertex_count(
            "/World/envs/env_2/Deformable/geometry/mesh",
            lookup,
            fallback=128,
        )
        == 69
    )
    assert resolve_deformable_vertex_count("/World/other", lookup, fallback=128) == 128


def test_deformable_root_path_lookup_and_resolve():
    """Root-path lookup rewrites env-relative matches onto the reported environment."""
    entries = [
        DeformableStageEntry(
            root_path="/World/envs/env_0/Deformable",
            sim_mesh_path="/World/envs/env_0/Deformable/geometry/mesh",
            vis_mesh_path="/World/envs/env_0/Deformable/geometry/vis_mesh",
            deformable_type="volume",
            vertex_count=69,
            vis_vertex_count=72,
        )
    ]
    lookup = build_deformable_root_path_lookup(entries)
    assert lookup["/World/envs/env_0/Deformable"] == "/World/envs/env_0/Deformable"
    assert lookup["/World/envs/env_0/Deformable/geometry/mesh"] == "/World/envs/env_0/Deformable"
    assert lookup["/World/envs/env_0/Deformable/geometry/vis_mesh"] == "/World/envs/env_0/Deformable"
    assert (
        resolve_deformable_root_path(
            "/World/envs/env_0/Deformable/geometry/mesh/points",
            lookup,
        )
        == "/World/envs/env_0/Deformable"
    )
    assert (
        resolve_deformable_root_path(
            "/World/envs/env_2/Deformable/geometry/mesh",
            lookup,
        )
        == "/World/envs/env_2/Deformable"
    )
    assert resolve_deformable_root_path("/World/other", lookup) == "/World/other"


def test_group_deformable_root_paths_for_views_collapses_replicated_env_assets():
    root_paths = [
        "/World/envs/env_0/ClothA",
        "/World/envs/env_1/ClothA",
        "/World/envs/env_0/SoftB",
    ]
    path_to_type = {
        "/World/envs/env_0/ClothA": "surface",
        "/World/envs/env_1/ClothA": "surface",
        "/World/envs/env_0/SoftB": "volume",
    }
    grouped = group_deformable_root_paths_for_views(root_paths, path_to_type)
    assert grouped["surface"] == (["/World/envs/env_*/ClothA"], [])
    assert grouped["volume"] == (["/World/envs/env_*/SoftB"], [])


def test_sort_deformable_entries_for_geometry_sync_orders_volume_before_surface():
    entries = [
        DeformableStageEntry(
            root_path="/World/envs/env_0/Cloth",
            sim_mesh_path="/World/envs/env_0/Cloth/sim",
            vis_mesh_path="/World/envs/env_0/Cloth/vis",
            deformable_type="surface",
            vertex_count=4,
            vis_vertex_count=4,
        ),
        DeformableStageEntry(
            root_path="/World/envs/env_0/Soft",
            sim_mesh_path="/World/envs/env_0/Soft/sim",
            vis_mesh_path="/World/envs/env_0/Soft/vis",
            deformable_type="volume",
            vertex_count=5,
            vis_vertex_count=5,
        ),
    ]
    ordered = sort_deformable_entries_for_geometry_sync(entries)
    assert [entry.root_path for entry in ordered] == [
        "/World/envs/env_0/Soft",
        "/World/envs/env_0/Cloth",
    ]
