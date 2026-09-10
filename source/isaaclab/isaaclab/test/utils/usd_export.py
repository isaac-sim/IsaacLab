# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent USD parser checks supplementing fresh-backend export tests."""

import numpy as np

from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade


def capture_physics_structure(stage: Usd.Stage) -> dict:
    """Capture parsed entity coverage, topology, collision geometry and filtering by prim identity.

    Runtime body poses, drives, limits and material coefficients are compared through backend
    views by the caller. Everything else exposed by the USD physics descriptors is retained here,
    including geometry dimensions, shape-to-body associations and joint attachment frames.
    """
    result = {}
    buffered = {"position", "rotation", "linearVelocity", "angularVelocity", "materials", "drive", "limit"}
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
    for kind, (paths, descriptions) in parsed.items():
        if kind in (UsdPhysics.ObjectType.Scene, UsdPhysics.ObjectType.RigidBodyMaterial):
            continue
        for path, description in zip(paths, descriptions):
            assert description.isValid, path
            for field in dir(description):
                if field.startswith("_") or field in buffered:
                    continue
                value = getattr(description, field)
                if not callable(value):
                    result[str(path), field] = _value(value)
            prim = stage.GetPrimAtPath(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if material:
                    for attribute in material.GetPrim().GetAttributes():
                        if "CombineMode" in attribute.GetName():
                            result[str(path), attribute.GetName()] = _value(attribute.Get())
    return result


def assert_physics_structure_equal(expected: dict, actual: dict) -> None:
    """Compare discrete structure exactly and geometric floating-point values with tolerances."""
    assert actual.keys() == expected.keys(), (actual.keys() - expected.keys(), expected.keys() - actual.keys())
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "f":
                np.testing.assert_allclose(other, value, rtol=1e-5, atol=1e-6, err_msg=str(key))
            else:
                np.testing.assert_array_equal(other, value, err_msg=str(key))
        elif isinstance(value, float):
            np.testing.assert_allclose(other, value, rtol=1e-5, atol=1e-6, err_msg=str(key))
        else:
            assert other == value, (key, other, value)


def _value(value):
    if isinstance(value, Sdf.Path):
        return str(value)
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, (Gf.Quatf, Gf.Quatd)):
        return np.array(Gf.Matrix3d(Gf.Quatd(value)))
    try:
        values = list(value)
    except TypeError:
        # USD enum values have stable symbolic names; descriptor subobjects expose properties.
        fields = {
            name: _value(getattr(value, name))
            for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name))
        }
        return fields if fields else str(value)
    if all(isinstance(v, (float, int)) for v in values):
        return np.asarray(values)
    return tuple(_value(v) for v in values)
