# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka pour hollow-cube bowl mesh (no simulator)."""

import ast
from pathlib import Path

import numpy as np
import pytest

from isaaclab_tasks.contrib.franka_pour import cube_bowl_mesh
from isaaclab_tasks.contrib.franka_pour.cube_bowl_mesh import cube_bowl_inner_bounds, make_cube_bowl_mesh

DIMS = dict(
    inner_width=0.037,
    inner_depth=0.037,
    cavity_depth=0.045,
    wall_thickness=0.009,
    bottom_thickness=0.009,
)


def test_returns_float32_flat_int32_arrays():
    v, f = make_cube_bowl_mesh(**DIMS)
    assert v.dtype == np.float32 and v.ndim == 2 and v.shape[1] == 3
    assert f.dtype == np.int32 and f.ndim == 1 and f.size % 3 == 0
    assert int(f.max()) < len(v)


def test_is_watertight_outward_manifold():
    # validate=True runs the task-local validator and raises if the shell is not a closed,
    # consistently-wound manifold.
    make_cube_bowl_mesh(**DIMS, validate=True)


def test_mesh_validation_is_task_local_and_rejects_open_shells():
    source_path = Path(cube_bowl_mesh.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("franka_scoop" in module for module in imported_modules)

    validator = getattr(cube_bowl_mesh, "_validate_closed_oriented_mesh", None)
    assert callable(validator)
    _, indices = make_cube_bowl_mesh(**DIMS, validate=False)
    with pytest.raises(RuntimeError, match="not watertight"):
        validator(indices[:-3], "Open cube bowl")


def test_outward_orientation_positive_signed_volume():
    v, f = make_cube_bowl_mesh(**DIMS)
    tris = v[f.reshape(-1, 3)]
    signed_vol = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum())
    assert signed_vol > 0.0


def test_outer_footprint_fits_gripper():
    v, _ = make_cube_bowl_mesh(**DIMS)
    outer = float(v[:, 0].max() - v[:, 0].min())
    assert abs(outer - (DIMS["inner_width"] + 2 * DIMS["wall_thickness"])) < 1e-6
    assert outer <= 0.06  # fits the ~0.08 m Franka opening with closure margin


def test_height_and_floor():
    v, _ = make_cube_bowl_mesh(**DIMS)
    assert abs(float(v[:, 2].min()) - 0.0) < 1e-6
    assert abs(float(v[:, 2].max()) - (DIMS["bottom_thickness"] + DIMS["cavity_depth"])) < 1e-6


def test_inner_bounds_inside_outer():
    lo, hi = cube_bowl_inner_bounds(
        DIMS["inner_width"], DIMS["inner_depth"], DIMS["cavity_depth"], DIMS["bottom_thickness"]
    )
    assert np.all(hi - lo > 0)
    assert abs(float(hi[0] - lo[0]) - DIMS["inner_width"]) < 1e-6
    assert abs(float(hi[1] - lo[1]) - DIMS["inner_depth"]) < 1e-6
    assert abs(float(lo[2]) - DIMS["bottom_thickness"]) < 1e-6
    assert abs(float(hi[2]) - (DIMS["bottom_thickness"] + DIMS["cavity_depth"])) < 1e-6
