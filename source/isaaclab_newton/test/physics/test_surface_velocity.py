# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle tests for Newton surface velocity."""

from __future__ import annotations

from types import SimpleNamespace

import isaaclab_newton.physics.surface_velocity as surface_module
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics.surface_velocity import compute_point_impulse

from isaaclab.physics import PhysicsEvent, SurfaceVelocitySpec

_BODY_PATTERN = r"(?:^|/)Cube_?[0-3](?:/|$)"


@wp.kernel
def _compute_test_impulses(
    target_velocity: wp.array(dtype=wp.vec3),
    normal_impulse: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    output[index] = compute_point_impulse(
        wp.vec3(0.0, 0.0, 1.0),
        normal_impulse[index],
        wp.vec3(),
        target_velocity[index],
        1.0,
        wp.mat33(),
        wp.vec3(),
        0.5,
        1.0,
    )


def _surface_spec(name: str = "Belt") -> SurfaceVelocitySpec:
    """Build one valid replicated test belt."""
    return SurfaceVelocitySpec(prim_path=f"{{ENV_REGEX_NS}}/{name}", velocity=0.35, friction_coefficient=0.5)


def test_point_impulse_tracks_velocity_and_respects_coulomb_limit() -> None:
    """Point traction reaches a small target but clamps large requests to ``mu * normal_impulse``."""
    target_velocity = wp.array([(0.25, 0.0, 0.0), (10.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
    normal_impulse = wp.array([2.0, 2.0], dtype=wp.float32, device="cpu")
    output = wp.zeros(2, dtype=wp.vec3, device="cpu")

    wp.launch(
        _compute_test_impulses,
        dim=2,
        inputs=[target_velocity, normal_impulse],
        outputs=[output],
        device="cpu",
    )

    np.testing.assert_allclose(output.numpy(), ((0.25, 0.0, 0.0), (1.0, 0.0, 0.0)), atol=1.0e-6)


class _FakeCallbackHandle:
    def __init__(self) -> None:
        self.deregister_count = 0

    def deregister(self) -> None:
        self.deregister_count += 1


class _FakeBinding:
    instances = []

    def __init__(self, model, contacts, **kwargs) -> None:
        self.model = model
        self.contacts = contacts
        self.kwargs = kwargs
        self.closed = False
        self._command_velocity_host = np.array([0.35, 0.35], dtype=np.float32)
        self._enabled_host = np.ones(2, dtype=np.int32)
        type(self).instances.append(self)

    def set_velocities(self, values) -> None:
        self._command_velocity_host = np.asarray(values, dtype=np.float32).copy()

    def set_enabled(self, values) -> None:
        self._enabled_host = np.asarray(values, dtype=np.int32).copy()

    def close(self) -> None:
        self.closed = True


def test_driver_requests_force_and_rebinds_on_solver_reinitialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver binds pre-capture and replaces model-owned buffers after a hard reset."""
    event_callbacks = []
    solver_callbacks = []
    unregistered_solver_callbacks = []
    requested_attributes = []
    callback_handle = _FakeCallbackHandle()
    _FakeBinding.instances = []

    def register_callback(cls, callback, event, order=0, name=None, wrap_weak_ref=True):
        event_callbacks.append((callback, event, name))
        return callback_handle

    monkeypatch.setattr(surface_module.NewtonManager, "register_callback", classmethod(register_callback))
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_solver_init_callback",
        classmethod(lambda cls, callback: solver_callbacks.append(callback)),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "unregister_solver_init_callback",
        classmethod(lambda cls, callback: unregistered_solver_callbacks.append(callback)),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "request_extended_contact_attribute",
        classmethod(lambda cls, attribute: requested_attributes.append(attribute)),
    )
    monkeypatch.setattr(surface_module, "_SurfaceVelocityBinding", _FakeBinding)

    driver = surface_module.SurfaceVelocity(num_envs=2, surface_specs=(_surface_spec(),), body_pattern=_BODY_PATTERN)

    assert driver.specs == (_surface_spec(),)
    assert driver.surfaces_per_env == 1
    assert driver.num_surfaces == 2
    assert driver.count == 2
    assert not driver.initialized

    assert [(event, name) for _, event, name in event_callbacks] == [
        (PhysicsEvent.MODEL_INIT, "surface_velocity_contact_attribute")
    ]
    event_callbacks[0][0](None)
    assert requested_attributes == ["force"]

    first_model, first_contacts = object(), object()
    solver_callbacks[0](first_model, first_contacts)
    first_binding = _FakeBinding.instances[-1]
    assert driver.initialized
    first_binding.set_velocities([0.2, -0.1])
    first_binding.set_enabled([1, 0])

    second_model, second_contacts = object(), object()
    solver_callbacks[0](second_model, second_contacts)
    second_binding = _FakeBinding.instances[-1]

    assert first_binding.closed
    assert second_binding.model is second_model
    assert second_binding.contacts is second_contacts
    np.testing.assert_allclose(second_binding._command_velocity_host, [0.2, -0.1])
    np.testing.assert_array_equal(second_binding._enabled_host, [1, 0])

    driver.close()
    driver.close()
    assert second_binding.closed
    assert unregistered_solver_callbacks == [solver_callbacks[0]]
    assert callback_handle.deregister_count == 1


def test_unbound_driver_rejects_control_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control methods are unavailable until the solver-init callback creates a binding."""
    callback_handle = _FakeCallbackHandle()
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_callback",
        classmethod(lambda cls, *args, **kwargs: callback_handle),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_solver_init_callback",
        classmethod(lambda cls, callback: None),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "unregister_solver_init_callback",
        classmethod(lambda cls, callback: None),
    )

    driver = surface_module.SurfaceVelocity(num_envs=1, surface_specs=(_surface_spec(),), body_pattern=_BODY_PATTERN)
    with pytest.raises(RuntimeError, match="not bound"):
        driver.set_velocities(0.2)
    driver.close()


def test_driver_rejects_invalid_specs_before_registering_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid descriptions cannot leave lifecycle callbacks behind."""
    registered = []
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_callback",
        classmethod(lambda cls, *args, **kwargs: registered.append(args)),
    )

    with pytest.raises(ValueError, match="At least one"):
        surface_module.SurfaceVelocity(num_envs=1, surface_specs=(), body_pattern=_BODY_PATTERN)
    with pytest.raises(TypeError, match="SurfaceVelocitySpec"):
        surface_module.SurfaceVelocity(num_envs=1, surface_specs=(object(),), body_pattern=_BODY_PATTERN)
    with pytest.raises(ValueError, match="unique"):
        surface_module.SurfaceVelocity(
            num_envs=1, surface_specs=(_surface_spec(), _surface_spec()), body_pattern=_BODY_PATTERN
        )
    with pytest.raises(ValueError, match="ancestors"):
        surface_module.SurfaceVelocity(
            num_envs=1,
            surface_specs=(
                SurfaceVelocitySpec(prim_path="{ENV_REGEX_NS}/Belt"),
                SurfaceVelocitySpec(prim_path="{ENV_REGEX_NS}/Belt/Child"),
            ),
            body_pattern=_BODY_PATTERN,
        )
    with pytest.raises(ValueError, match="explicit positive radius"):
        surface_module.SurfaceVelocity(
            num_envs=1,
            surface_specs=(SurfaceVelocitySpec(prim_path="{ENV_REGEX_NS}/Curve", curved=True),),
            body_pattern=_BODY_PATTERN,
        )
    with pytest.raises(ValueError, match="Replicated conveyor environments"):
        surface_module.SurfaceVelocity(
            num_envs=2,
            surface_specs=(SurfaceVelocitySpec(prim_path="/World/Shared/Belt"),),
            body_pattern=_BODY_PATTERN,
        )
    with pytest.raises(ValueError, match="env_path_format"):
        surface_module.SurfaceVelocity(
            num_envs=1,
            surface_specs=(_surface_spec(),),
            body_pattern=_BODY_PATTERN,
            env_path_format="/World/envs/env_.*",
        )

    assert registered == []


def test_belt_paths_are_exact_and_environment_scoped() -> None:
    """A descriptor cannot bind a same-named shape outside the replicated environment root."""
    resolve = surface_module._resolve_belt_prim_path

    assert resolve("{ENV_REGEX_NS}/Belt", "/World/envs/env_{}", 0) == "/World/envs/env_0/Belt"
    assert resolve("{ENV_REGEX_NS}/Nested/Belt", "/World/envs/env_{}", 123) == ("/World/envs/env_123/Nested/Belt")
    assert resolve("/World/Shared/Belt", "/World/envs/env_{}", 7) == "/World/Shared/Belt"
    belongs = surface_module._shape_belongs_to_prim
    assert belongs("/World/envs/env_0/Belt/geometry/mesh", "/World/envs/env_0/Belt")
    assert not belongs("/World/props/Belt/geometry/mesh", "/World/envs/env_0/Belt")
    assert not belongs("/World/envs/env_0/Nested/Belt", "/World/envs/env_0/Belt")


def test_driver_cleans_up_model_callback_when_solver_registration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lifecycle registration failure cannot leave a partially active driver."""
    callback_handle = _FakeCallbackHandle()
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_callback",
        classmethod(lambda cls, *args, **kwargs: callback_handle),
    )

    def fail_registration(cls, callback):
        raise RuntimeError("solver callback unavailable")

    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_solver_init_callback",
        classmethod(fail_registration),
    )

    with pytest.raises(RuntimeError, match="solver callback unavailable"):
        surface_module.SurfaceVelocity(num_envs=1, surface_specs=(_surface_spec(),), body_pattern=_BODY_PATTERN)

    assert callback_handle.deregister_count == 1


def test_binding_uses_deterministic_environment_major_belt_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Newton discovery order cannot reorder commands or encoder rows after a rebuild."""
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_state_force_callback",
        classmethod(lambda cls, callback: None),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "register_post_solver_substep_callback",
        classmethod(lambda cls, callback: None),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "unregister_state_force_callback",
        classmethod(lambda cls, callback: None),
    )
    monkeypatch.setattr(
        surface_module.NewtonManager,
        "unregister_post_solver_substep_callback",
        classmethod(lambda cls, callback: None),
    )

    shape_labels = (
        "/World/envs/env_1/BeltB",
        "/World/envs/env_0/BeltA",
        "/World/envs/env_1/BeltA",
        "/World/envs/env_0/BeltB",
    )
    shape_count = len(shape_labels)
    identity = wp.transform(wp.vec3(), wp.quat_identity())
    model = SimpleNamespace(
        world_count=2,
        device="cpu",
        shape_count=shape_count,
        shape_label=shape_labels,
        shape_body=wp.full(shape_count, -1, dtype=wp.int32, device="cpu"),
        shape_world=wp.array([1, 0, 1, 0], dtype=wp.int32, device="cpu"),
        shape_transform=wp.array([identity] * shape_count, dtype=wp.transform, device="cpu"),
        body_count=2,
        body_label=("/World/envs/env_0/Cube0", "/World/envs/env_1/Cube0"),
        body_world=wp.array([0, 1], dtype=wp.int32, device="cpu"),
        body_com=wp.zeros(2, dtype=wp.vec3, device="cpu"),
        body_inv_mass=wp.ones(2, dtype=wp.float32, device="cpu"),
        body_inv_inertia=wp.array([wp.mat33(1.0)] * 2, dtype=wp.mat33, device="cpu"),
    )
    contact_capacity = 4
    contacts = SimpleNamespace(
        rigid_contact_max=contact_capacity,
        force=wp.zeros(contact_capacity, dtype=wp.spatial_vector, device="cpu"),
        rigid_contact_shape0=wp.full(contact_capacity, -1, dtype=wp.int32, device="cpu"),
        rigid_contact_shape1=wp.full(contact_capacity, -1, dtype=wp.int32, device="cpu"),
        rigid_contact_normal=wp.zeros(contact_capacity, dtype=wp.vec3, device="cpu"),
        rigid_contact_point0=wp.zeros(contact_capacity, dtype=wp.vec3, device="cpu"),
        rigid_contact_point1=wp.zeros(contact_capacity, dtype=wp.vec3, device="cpu"),
        rigid_contact_count=wp.zeros(1, dtype=wp.int32, device="cpu"),
    )
    specs = (
        SurfaceVelocitySpec(
            prim_path="{ENV_REGEX_NS}/BeltA",
            velocity=0.1,
            friction_coefficient=0.4,
            contact_threshold=0.98,
        ),
        SurfaceVelocitySpec(
            prim_path="{ENV_REGEX_NS}/BeltB",
            velocity=-0.2,
            enabled=False,
            friction_coefficient=0.6,
            contact_threshold=0.99,
        ),
    )

    binding = surface_module._SurfaceVelocityBinding(
        model=model,
        contacts=contacts,
        num_envs=2,
        surface_specs=specs,
        body_pattern=_BODY_PATTERN,
        body_count_per_env=1,
    )
    try:
        assert binding.surface_paths == (
            "/World/envs/env_0/BeltA",
            "/World/envs/env_0/BeltB",
            "/World/envs/env_1/BeltA",
            "/World/envs/env_1/BeltB",
        )
        np.testing.assert_array_equal(binding._shape_conveyor.numpy(), [3, 0, 2, 1])
        np.testing.assert_array_equal(binding._conveyor_world.numpy(), [0, 0, 1, 1])
        np.testing.assert_allclose(binding._command_velocity_host, [0.1, -0.2, 0.1, -0.2])
        np.testing.assert_array_equal(binding._enabled_host, [1, 0, 1, 0])
        np.testing.assert_allclose(binding._friction.numpy(), [0.4, 0.6, 0.4, 0.6])
        np.testing.assert_allclose(binding._threshold.numpy(), [0.98, 0.99, 0.98, 0.99])
        np.testing.assert_array_equal(binding.get_enabled(indices=[3, 0]).numpy(), [0, 1])
    finally:
        binding.close()
