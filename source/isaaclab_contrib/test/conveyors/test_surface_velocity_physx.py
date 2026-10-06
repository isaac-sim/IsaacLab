# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light tests for native PhysX surface velocity."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

import isaaclab_contrib.conveyors.physx as surface_module
from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec


class _FakeWriter:
    """Record authored states without importing USD or PhysX schemas."""

    def __init__(self) -> None:
        self.writes: list[tuple[int, bool, surface_module.PhysxSurfaceVelocityTwist]] = []
        self.close_count = 0

    def write(self, index: int, *, enabled: bool, twist: surface_module.PhysxSurfaceVelocityTwist) -> None:
        self.writes.append((index, enabled, twist))

    def close(self) -> None:
        self.close_count += 1


def _surface_spec(name: str = "Belt", **kwargs) -> SurfaceVelocitySpec:
    """Return one replicated belt description for facade tests."""
    return SurfaceVelocitySpec(prim_path=f"{{ENV_REGEX_NS}}/{name}", velocity=0.4, **kwargs)


def test_twist_conversion_normalizes_straight_direction() -> None:
    """Straight surface speed is independent of the authored direction magnitude."""
    spec = _surface_spec(direction=(3.0, 4.0, 0.0))

    twist = surface_module.compute_surface_velocity_twist(spec, velocity=2.0)

    np.testing.assert_allclose(twist.linear_velocity, (1.2, 1.6, 0.0))
    assert twist.angular_velocity_deg == (0.0, 0.0, 0.0)


def test_twist_conversion_uses_degrees_and_compensates_curved_pivot() -> None:
    """Curved belts rotate about their local pivot rather than the rigid-body origin."""
    spec = _surface_spec(
        direction=(0.0, 0.0, 2.0),
        curved=True,
        radius=2.0,
        pivot_point=(2.0, 0.0, 0.0),
    )

    twist = surface_module.compute_surface_velocity_twist(spec, velocity=math.pi)

    np.testing.assert_allclose(twist.angular_velocity_deg, (0.0, 0.0, 90.0), atol=1.0e-12)
    np.testing.assert_allclose(twist.linear_velocity, (0.0, -math.pi, 0.0), atol=1.0e-12)


def test_curved_twist_requires_an_explicit_radius() -> None:
    """A native angular rate cannot be inferred from unspecified task geometry."""
    spec = _surface_spec(curved=True)

    with pytest.raises(ValueError, match="positive radius"):
        surface_module.compute_surface_velocity_twist(spec)


def test_paths_are_resolved_in_environment_major_order() -> None:
    """Runtime rows stay deterministic across stage discovery ordering."""
    specs = (_surface_spec("BeltA"), _surface_spec("Nested/BeltB"))

    paths = surface_module.resolve_surface_velocity_paths(2, specs)

    assert paths == (
        "/World/envs/env_0/BeltA",
        "/World/envs/env_0/Nested/BeltB",
        "/World/envs/env_1/BeltA",
        "/World/envs/env_1/Nested/BeltB",
    )
    with pytest.raises(ValueError, match="require every belt"):
        surface_module.resolve_surface_velocity_paths(2, (SurfaceVelocitySpec(prim_path="/World/Shared/Belt"),))


def test_facade_ramps_playback_integrates_encoders_and_preserves_commands_on_reset() -> None:
    """Full resets restart playback without erasing policy-visible command state."""
    writer = _FakeWriter()
    facade = surface_module.SurfaceVelocity(2, (_surface_spec(),), writer=writer)

    assert facade.prim_paths == ("/World/envs/env_0/Belt", "/World/envs/env_1/Belt")
    assert facade.num_surfaces == facade.count == 2
    assert [record[2].linear_velocity for record in writer.writes] == [(0.0, 0.0, 0.0)] * 2

    facade.update(0.25)
    np.testing.assert_allclose([record[2].linear_velocity[0] for record in writer.writes[-2:]], (0.1, 0.1))
    np.testing.assert_allclose(facade.get_encoder_positions(), (0.1, 0.1))

    facade.set_velocities(torch.tensor([0.8], requires_grad=True), indices=torch.tensor([0]))
    facade.set_enabled(False, indices=[1])
    facade.update(0.25)
    np.testing.assert_allclose(facade.get_commanded_velocities(), (0.8, 0.4))
    np.testing.assert_allclose(facade.get_velocities(), (0.8, 0.0))
    np.testing.assert_allclose(facade.get_encoder_positions(), (0.3, 0.1))

    facade.reset(env_ids=torch.tensor([0]))
    np.testing.assert_allclose(facade.get_encoder_positions(), (0.0, 0.1))
    facade.reset(env_ids=[1, 0])
    np.testing.assert_allclose(facade.get_encoder_positions(), (0.0, 0.0))
    np.testing.assert_allclose(facade.get_commanded_velocities(), (0.8, 0.4))
    assert facade.get_enabled().tolist() == [1, 0]
    np.testing.assert_allclose(writer.writes[-2][2].linear_velocity, (0.0, 0.0, 0.0))

    facade.close()
    facade.close()
    assert writer.close_count == 1
    assert [(index, enabled) for index, enabled, _ in writer.writes[-2:]] == [(0, False), (1, False)]


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

    surface_module.apply_surface_velocity_api(prim, _surface_spec(), velocity_scale=0.0)

    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    assert rigid_body.GetRigidBodyEnabledAttr().Get() is True
    assert rigid_body.GetKinematicEnabledAttr().Get() is True
    schema.PhysxSurfaceVelocityAPI.Apply.assert_called_once_with(prim)
    surface_api.CreateSurfaceVelocityLocalSpaceAttr().Set.assert_called_once_with(True)
    assert surface_api.CreateSurfaceVelocityEnabledAttr().Set.call_args_list == [call(False), call(True)]
    surface_api.CreateSurfaceVelocityAttr().Set.assert_called_once_with(Gf.Vec3f(0.0))
    surface_api.CreateSurfaceAngularVelocityAttr().Set.assert_called_once_with(Gf.Vec3f(0.0))
