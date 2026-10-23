# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light tests for native PhysX surface velocity."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

import isaaclab_contrib.conveyors.physx as surface_module
from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec


class _Attribute:
    def __init__(self):
        self.values = []

    def Set(self, value):
        self.values.append(value)


def _surface_spec(name: str = "Belt", velocity: float = 0.4, **kwargs) -> SurfaceVelocitySpec:
    return SurfaceVelocitySpec(prim_path=f"{{ENV_REGEX_NS}}/{name}", velocity=velocity, **kwargs)


def test_twist_conversion_normalizes_straight_direction() -> None:
    """Straight surface speed is independent of the authored direction magnitude."""
    twist = surface_module.compute_surface_velocity_twist(_surface_spec(direction=(3.0, 4.0, 0.0), velocity=-2.0))
    np.testing.assert_allclose(twist.linear_velocity, (-1.2, -1.6, 0.0))
    assert twist.angular_velocity_deg == (0.0, 0.0, 0.0)


def test_twist_conversion_uses_degrees_and_compensates_curved_pivot() -> None:
    """Curved belts rotate about their local pivot rather than the rigid-body origin."""
    spec = _surface_spec(
        direction=(0.0, 0.0, 2.0), curved=True, radius=2.0, pivot_point=(2.0, 0.0, 0.0), velocity=math.pi
    )
    twist = surface_module.compute_surface_velocity_twist(spec)
    np.testing.assert_allclose(twist.angular_velocity_deg, (0.0, 0.0, 90.0), atol=1.0e-12)
    np.testing.assert_allclose(twist.linear_velocity, (0.0, -math.pi, 0.0), atol=1.0e-12)


def test_ramp_reset_and_callback_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full resets restart the ramp; partial resets and steady playback avoid USD writes."""
    attributes = {f"/World/envs/env_{i}/Belt": tuple(_Attribute() for _ in range(3)) for i in range(2)}
    monkeypatch.setattr(surface_module, "_surface_attributes", lambda prim, **kwargs: attributes[prim])
    register = Mock(return_value=Mock())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_physx.physics.physx_manager",
        SimpleNamespace(
            IsaacEvents=SimpleNamespace(POST_PHYSICS_STEP="step"),
        ),
    )
    adapter = surface_module.SurfaceVelocity(
        2,
        (_surface_spec(),),
        stage=SimpleNamespace(GetPrimAtPath=lambda path: path),
        manager=SimpleNamespace(register_callback=register),
    )
    adapter.start()
    adapter.start()
    register.assert_called_once()
    update = register.call_args.args[0]
    for enabled, linear, angular in attributes.values():
        assert enabled.values[-1] is True and tuple(linear.values[-1]) == (0, 0, 0)
    update(0.25)
    for _, linear, _ in attributes.values():
        np.testing.assert_allclose(linear.values[-1], (0.1, 0, 0))
    writes = len(attributes["/World/envs/env_0/Belt"][1].values)
    adapter.reset(torch.tensor([0]))
    assert len(attributes["/World/envs/env_0/Belt"][1].values) == writes
    update(1.0)
    writes += 1
    update(1.0)
    assert len(attributes["/World/envs/env_0/Belt"][1].values) == writes
    for _, linear, _ in attributes.values():
        np.testing.assert_allclose(linear.values[-1], (0.4, 0, 0))
    adapter.reset([1, 0])
    for _, linear, _ in attributes.values():
        assert tuple(linear.values[-1]) == (0, 0, 0)
    update(0.25)
    for _, linear, _ in attributes.values():
        np.testing.assert_allclose(linear.values[-1], (0.1, 0, 0))
    adapter.reset()
    adapter.close()
    adapter.close()
    register.return_value.deregister.assert_called_once()
    for enabled, linear, angular in attributes.values():
        assert enabled.values[-1] is False
        assert tuple(linear.values[-1]) == tuple(angular.values[-1]) == (0, 0, 0)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.start()


def test_authoring_helper_applies_kinematic_local_surface_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Author a real kinematic USD body; mock only the Kit-dependent PhysX API."""
    import pxr
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Belt", "Xform")
    surface_api = Mock()
    # HasAPI needs a registered USD type. Substitute only the unavailable PhysX schema.
    schema = SimpleNamespace(PhysxSurfaceVelocityAPI=UsdPhysics.CollisionAPI)
    monkeypatch.setattr(UsdPhysics.CollisionAPI, "Apply", Mock(return_value=surface_api))
    monkeypatch.setattr(pxr, "PhysxSchema", schema, raising=False)

    surface_module.apply_surface_velocity_api(prim)

    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    assert rigid_body.GetRigidBodyEnabledAttr().Get() is True
    assert rigid_body.GetKinematicEnabledAttr().Get() is True
    schema.PhysxSurfaceVelocityAPI.Apply.assert_called_once_with(prim)
    surface_api.CreateSurfaceVelocityLocalSpaceAttr().Set.assert_called_once_with(True)
    assert surface_api.CreateSurfaceVelocityEnabledAttr().Set.call_args_list == [call(False), call(True)]
    surface_api.CreateSurfaceVelocityAttr().Set.assert_called_once_with(Gf.Vec3f(0.0))
    surface_api.CreateSurfaceAngularVelocityAttr().Set.assert_called_once_with(Gf.Vec3f(0.0))
