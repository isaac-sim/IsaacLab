# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light tests for native PhysX surface velocity."""

from __future__ import annotations

import math
import sys
from types import ModuleType

import isaaclab_physx.physics.surface_velocity as surface_module
import numpy as np
import pytest
import torch

from isaaclab.physics import SurfaceVelocitySpec


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
    omega_rad = np.radians(twist.angular_velocity_deg)
    point = np.asarray((3.0, 0.0, 0.0))
    actual_at_point = np.asarray(twist.linear_velocity) + np.cross(omega_rad, point)
    expected_at_point = np.cross(omega_rad, point - np.asarray(spec.pivot_point))
    np.testing.assert_allclose(actual_at_point, expected_at_point)


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

    facade.set_velocities(0.8, indices=[0])
    facade.set_enabled(False, indices=[1])
    facade.update(0.25)
    np.testing.assert_allclose(facade.get_commanded_velocities(), (0.8, 0.4))
    np.testing.assert_allclose(facade.get_velocities(), (0.8, 0.0))
    np.testing.assert_allclose(facade.get_encoder_positions(), (0.3, 0.1))

    facade.reset(env_ids=[0])
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


def test_facade_accepts_torch_control_and_reset_indices() -> None:
    """Normal Isaac Lab tensor selectors are copied to host before NumPy validation."""
    facade = surface_module.SurfaceVelocity(2, (_surface_spec(),), writer=_FakeWriter())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    facade.set_velocities(torch.tensor([0.6], device=device), indices=torch.tensor([1], device=device))
    facade.update(0.25)
    facade.reset(env_ids=torch.tensor([1], device=device))

    np.testing.assert_allclose(facade.get_commanded_velocities(), (0.4, 0.6))
    np.testing.assert_allclose(facade.get_encoder_positions(), (0.1, 0.0))
    facade.close()


def test_authoring_helper_applies_kinematic_local_surface_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy authoring seam applies both schemas and authors an initially stopped belt."""

    class FakeAttribute:
        def __init__(self) -> None:
            self.value = None

        def Set(self, value) -> bool:
            self.value = value
            return True

        def Get(self):
            return self.value

    class FakePrim:
        def __init__(self) -> None:
            self.apis = set()
            self.attributes = {}

        def IsValid(self) -> bool:
            return True

        def HasAPI(self, api_type) -> bool:
            return api_type in self.apis

        def GetPath(self) -> str:
            return "/World/Belt"

    class FakeRigidBodyAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        @classmethod
        def Apply(cls, prim: FakePrim):
            prim.apis.add(cls)
            return cls(prim)

        def CreateRigidBodyEnabledAttr(self) -> FakeAttribute:
            return self.prim.attributes.setdefault("rigid_enabled", FakeAttribute())

        def CreateKinematicEnabledAttr(self) -> FakeAttribute:
            return self.prim.attributes.setdefault("kinematic", FakeAttribute())

        def GetKinematicEnabledAttr(self) -> FakeAttribute:
            return self.prim.attributes.setdefault("kinematic", FakeAttribute())

    class FakeSurfaceAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        @classmethod
        def Apply(cls, prim: FakePrim):
            prim.apis.add(cls)
            return cls(prim)

        def _attribute(self, name: str) -> FakeAttribute:
            return self.prim.attributes.setdefault(name, FakeAttribute())

        def CreateSurfaceVelocityLocalSpaceAttr(self) -> FakeAttribute:
            return self._attribute("local_space")

        def CreateSurfaceVelocityEnabledAttr(self) -> FakeAttribute:
            return self._attribute("surface_enabled")

        def CreateSurfaceVelocityAttr(self) -> FakeAttribute:
            return self._attribute("linear")

        def CreateSurfaceAngularVelocityAttr(self) -> FakeAttribute:
            return self._attribute("angular")

    fake_pxr = ModuleType("pxr")
    fake_pxr.Gf = type("FakeGf", (), {"Vec3f": staticmethod(lambda *values: tuple(values))})
    fake_pxr.PhysxSchema = type("FakePhysxSchema", (), {"PhysxSurfaceVelocityAPI": FakeSurfaceAPI})
    fake_pxr.UsdPhysics = type("FakeUsdPhysics", (), {"RigidBodyAPI": FakeRigidBodyAPI})
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    prim = FakePrim()

    surface_module.apply_surface_velocity_api(prim, _surface_spec(), velocity_scale=0.0)

    assert FakeRigidBodyAPI in prim.apis
    assert FakeSurfaceAPI in prim.apis
    assert prim.attributes["rigid_enabled"].value is True
    assert prim.attributes["kinematic"].value is True
    assert prim.attributes["local_space"].value is True
    assert prim.attributes["surface_enabled"].value is True
    assert prim.attributes["linear"].value == (0.0, 0.0, 0.0)
    assert prim.attributes["angular"].value == (0.0, 0.0, 0.0)
