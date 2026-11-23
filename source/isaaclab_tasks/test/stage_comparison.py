# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural USD comparison for registered-task snapshots."""

import re

import numpy as np

from pxr import Usd, UsdGeom


def compare_golden_stage(golden_path: str, result_path: str) -> list[str]:
    """Compare composed prim inventories/types and world transforms, ignoring generated product names."""
    snapshots = []
    for filename in (golden_path, result_path):
        stage = Usd.Stage.Open(filename)
        cache = UsdGeom.XformCache()
        products = sorted(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if re.fullmatch(r"/Render/OmniverseKit/HydraTextures/(Replicator|rp_[0-9a-f]{32})", str(prim.GetPath()))
        )
        structure, transforms = {}, {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for index, product in enumerate(products):
                if path == product or path.startswith(product + "/"):
                    path = product.rsplit("/", 1)[0] + f"/rp_canonical_{index}" + path[len(product) :]
                    break
            structure[path] = prim.GetTypeName()
            if UsdGeom.Xformable(prim):
                transforms[path] = np.asarray(cache.GetLocalToWorldTransform(prim))
        snapshots.append((structure, transforms))
    (expected, poses), (actual, actual_poses) = snapshots
    problems = [f"added prim {path}" for path in actual.keys() - expected.keys()]
    problems += [f"removed prim {path}" for path in expected.keys() - actual.keys()]
    for path in expected.keys() & actual.keys():
        if expected[path] != actual[path]:
            problems.append(f"type changed {path}: {expected[path]} -> {actual[path]}")
    for path in poses.keys() & actual_poses.keys():
        if not np.allclose(poses[path], actual_poses[path], rtol=1e-4, atol=1e-5):
            problems.append(f"transform {path} differs")
    return sorted(problems)
