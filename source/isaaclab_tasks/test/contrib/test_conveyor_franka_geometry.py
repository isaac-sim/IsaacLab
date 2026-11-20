# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-face orientation for the contributed conveyor task."""

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorFrankaSceneCfg

pytestmark = pytest.mark.kitless


def test_belt_collision_top_faces_point_upward():
    """Each authored belt binds to a collider, and curved decks support parcels from above."""
    cfg = ConveyorFrankaSceneCfg(num_envs=1, env_spacing=3.0)
    stage = Usd.Stage.Open(cfg.racetracks.spawn.usd_path)
    for spec in cfg.build_conveyor_belt_specs():
        section = stage.GetPrimAtPath(spec.prim_path.removeprefix("{ENV_REGEX_NS}"))
        (collider,) = (prim for prim in Usd.PrimRange(section) if prim.HasAPI(UsdPhysics.CollisionAPI))
        if not spec.curved:
            continue
        mesh = UsdGeom.Mesh(collider)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        assert set(counts) == {3}
        top_z = max(point[2] for point in points)
        top_faces = []
        for start in range(0, len(indices), 3):
            face = tuple(points[index] for index in indices[start : start + 3])
            if all(point[2] == top_z for point in face):
                top_faces.append(face)
        assert top_faces
        for a, b, c in top_faces:
            cross_z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            assert cross_z > 0.0
