# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared rigid-object and rigid-object-collection contracts across the production backends.

Both asset families run on mocked backend views, so these cases need neither Isaac Sim nor a GPU simulation. A rigid
object exposes root quantities of shape ``(N,)`` and body quantities of shape ``(N, 1)``; a collection exposes body
quantities of shape ``(N, B)``. Bookkeeping runs on CPU; getters and writers run on every test device because PhysX
stages writes through pinned CPU buffers on CUDA.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets import BaseRigidObjectCollectionData, BaseRigidObjectData
from isaaclab.utils.warp import ProxyArray

from .rigid_object_factory import get_rigid_object, get_rigid_object_collection

pytestmark = pytest.mark.integration

# Distinct instance and body counts make swapped axes visible.
_NUM_INSTANCES, _NUM_BODIES = 2, 3
_ASSETS = ["object", "collection"]


def _create(asset: str, backend: str, monkeypatch: pytest.MonkeyPatch, device: str = "cpu", num_instances: int = 2):
    """Create a mocked asset and return it with its raw backend view, its top-level prefix, and its body count."""
    if asset == "object":
        obj, raw = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
        return obj, raw, "root", 1
    obj, raw = get_rigid_object_collection(backend, num_instances, _NUM_BODIES, device, monkeypatch=monkeypatch)
    return obj, raw, "body", _NUM_BODIES


def _payload(shape: tuple[int, ...], trailing: tuple[int, ...], offset: float = 0.0, device: str = "cpu"):
    """Return per-element distinct values; 7-wide transforms carry a 90-degree rotation about Z."""
    data = torch.arange(np.prod(shape + trailing), dtype=torch.float32, device=device).reshape(shape + trailing)
    data += 1.25 + offset
    if trailing == (7,):
        data[..., 3:5] = 0.0
        data[..., 5:7] = 2.0**-0.5
    return data


def _select_last(values: torch.Tensor, selection: str, per_body: bool) -> tuple[torch.Tensor, dict]:
    """Return the payload and selectors that write the last environment (and last body) of ``values``.

    Selecting the last rather than the first element keeps identity or transposed routing from passing.
    """
    device = values.device
    if selection == "index":
        selected = values[-1:, -1:] if per_body else values[-1:]
        selectors = {"env_ids": torch.tensor([values.shape[0] - 1], dtype=torch.int32, device=device)}
        if per_body:
            selectors["body_ids"] = torch.tensor([values.shape[1] - 1], dtype=torch.int32, device=device)
        return selected, selectors
    masks = [wp.array(np.arange(n) == n - 1, dtype=wp.bool, device=str(device)) for n in values.shape[:2]]
    selectors = {"env_mask": masks[0]}
    if per_body:
        selectors["body_mask"] = masks[1]
    return values, selectors


# ---------------------------------------------------------------------------
# Bookkeeping, finders, and data properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset", _ASSETS)
def test_counts_names_finders_and_index_resolution(monkeypatch, backend, asset):
    obj, _, _, num_bodies = _create(asset, backend, monkeypatch, num_instances=4)

    assert obj.num_instances == 4
    assert obj.num_bodies == num_bodies
    assert len(obj.body_names) == num_bodies and all(isinstance(name, str) for name in obj.body_names)
    assert isinstance(obj.data, BaseRigidObjectData if asset == "object" else BaseRigidObjectCollectionData)

    # The legacy return mode is a list for rigid objects and an int32 tensor for collections.
    indices, names = obj.find_bodies(".*")
    proxy, proxy_names = obj.find_bodies(".*", as_proxy=True)
    assert isinstance(indices, list if asset == "object" else torch.Tensor)
    assert list(indices) == proxy.torch.tolist() == list(range(num_bodies))
    assert names == proxy_names == obj.body_names
    assert proxy is obj.find_bodies(".*", as_proxy=True)[0]
    assert proxy.dtype == wp.int32 and str(proxy.device) == obj.device
    assert obj.find_bodies(obj.body_names[-1])[1] == [obj.body_names[-1]]
    if asset == "collection":
        with pytest.warns(DeprecationWarning):
            assert obj.find_objects(".*", as_proxy=True)[0] is proxy
        assert obj._resolve_body_ids(torch.arange(3, dtype=torch.int32)[:2]).shape[0] == 2

    # Environment selections resolve to strided views of the cached indices without copying.
    env_ids = torch.arange(4, dtype=torch.int32)
    assert obj._resolve_env_ids(env_ids[:2]).shape[0] == 2
    cached = wp.to_torch(obj._ALL_INDICES if asset == "object" else obj._ALL_ENV_INDICES)
    for selection in (slice(None), slice(1, None, 2), slice(0, 0)):
        resolved = wp.to_torch(obj._resolve_env_ids(selection))
        torch.testing.assert_close(resolved, cached[selection])
        assert resolved.data_ptr() == cached[selection].data_ptr()
        assert resolved.stride() == cached[selection].stride()


def _expected_properties(top: str) -> dict[str, type]:
    """Return the public data properties and their dtypes for an asset whose top-level prefix is ``top``."""
    frame_quantities = {
        "pose_w": wp.transformf,
        "vel_w": wp.spatial_vectorf,
        "pos_w": wp.vec3f,
        "quat_w": wp.quatf,
        "lin_vel_w": wp.vec3f,
        "ang_vel_w": wp.vec3f,
    }
    properties = {
        f"{prefix}_{frame}_{quantity}": dtype
        for prefix in {top, "body"}
        for frame in ("link", "com")
        for quantity, dtype in frame_quantities.items()
    }
    properties |= {f"{top}_{frame}_{axis}_vel_b": wp.vec3f for frame in ("link", "com") for axis in ("lin", "ang")}
    return properties | {
        "projected_gravity_b": wp.vec3f,
        "heading_w": wp.float32,
        "body_com_acc_w": wp.spatial_vectorf,
        "body_com_lin_acc_w": wp.vec3f,
        "body_com_ang_acc_w": wp.vec3f,
        "body_com_pose_b": wp.transformf,
        "body_com_pos_b": wp.vec3f,
        "body_com_quat_b": wp.quatf,
        "body_mass": wp.float32,
        "body_inertia": wp.float32,
        f"default_{top}_pose": wp.transformf,
        f"default_{top}_vel": wp.spatial_vectorf,
    }


def _expected_aliases(top: str) -> dict[str, str]:
    """Return the shorthand data properties and the canonical properties they alias."""
    aliases = {
        "body_acc_w": "body_com_acc_w",
        "body_lin_acc_w": "body_com_lin_acc_w",
        "body_ang_acc_w": "body_com_ang_acc_w",
        "com_pos_b": "body_com_pos_b",
        "com_quat_b": "body_com_quat_b",
    }
    for prefix in {top, "body"}:
        aliases |= {f"{prefix}_{quantity}_w": f"{prefix}_link_{quantity}_w" for quantity in ("pose", "pos", "quat")}
        aliases |= {
            f"{prefix}_{quantity}_w": f"{prefix}_com_{quantity}_w" for quantity in ("vel", "lin_vel", "ang_vel")
        }
    return aliases


@pytest.mark.parametrize("asset", _ASSETS)
def test_data_properties_shapes_aliases_and_views(monkeypatch, backend, device, asset):
    obj, _, top, num_bodies = _create(asset, backend, monkeypatch, device)
    data = obj.data
    data.update(dt=0.01)
    if backend == "newton" and asset == "object":
        # Derived quantities stay unallocated until first read.
        for name in ("root_link_vel_w", "root_com_pose_w", "body_com_pose_b", "projected_gravity_b", "heading_w"):
            assert getattr(data, "_" + name).data is None

    for name, dtype in _expected_properties(top).items():
        value = getattr(data, name)
        per_instance = asset == "object" and not name.startswith("body")
        shape = (_NUM_INSTANCES,) if per_instance else (_NUM_INSTANCES, num_bodies)
        assert isinstance(value, ProxyArray), name
        assert (value.shape, value.dtype) == (shape + ((9,) if name == "body_inertia" else ()), dtype), name

    # Random mock state makes link and COM quantities differ, so a retargeted alias fails.
    for alias, canonical in _expected_aliases(top).items():
        alias_value, canonical_value = getattr(data, alias), getattr(data, canonical)
        assert (alias_value.shape, alias_value.dtype) == (canonical_value.shape, canonical_value.dtype), alias
        assert torch.equal(alias_value.torch, canonical_value.torch), alias

    # Position, orientation, and velocity components are zero-copy views of the packed pose and velocity.
    for frame in {f"{top}_link", f"{top}_com", "body_link", "body_com"}:
        for quantity, components in (("pose", ("pos", "quat")), ("vel", ("lin_vel", "ang_vel"))):
            packed = getattr(data, f"{frame}_{quantity}_w").torch
            for component, expected in zip(components, (packed[..., :3], packed[..., 3:]), strict=True):
                view = getattr(data, f"{frame}_{component}_w").torch
                torch.testing.assert_close(view, expected)
                assert (view.data_ptr(), view.stride()) == (expected.data_ptr(), expected.stride()), component


@pytest.mark.parametrize("trigger", ["pose", "velocity", "coms_index", "coms_mask"])
@pytest.mark.parametrize("asset", _ASSETS)
def test_writes_invalidate_dependent_caches(monkeypatch, backend, asset, trigger):
    obj, _, top, num_bodies = _create(asset, backend, monkeypatch)
    data = obj.data
    data.update(dt=0.01)
    body_frame_velocities = [f"{top}_{frame}_{axis}_vel_b" for frame in ("link", "com") for axis in ("lin", "ang")]
    stale = {
        "pose": [f"{top}_link_vel_w", "projected_gravity_b", "heading_w", *body_frame_velocities],
        "velocity": body_frame_velocities,
    }.get(trigger)
    if stale is None:
        states = [f"{top}_state_w", f"{top}_link_state_w", f"{top}_com_state_w"]
        stale = [f"{top}_com_pose_w", f"{top}_link_vel_w", *body_frame_velocities, *states]
        stale.append("body_com_pose_b" if backend == "newton" else f"{top}_com_vel_w")
    # Read each property so lazy caches allocate, then mark it current.
    buffers = {}
    for name in stale:
        getattr(data, name)
        buffers[name] = getattr(data, "_" + name)
        buffers[name].timestamp = data._sim_timestamp
    body_velocity = data.body_link_vel_w

    shape = (_NUM_INSTANCES,) if asset == "object" else (_NUM_INSTANCES, num_bodies)
    pose_kwarg, velocity_kwarg = (
        ("root_pose", "root_velocity") if asset == "object" else ("body_poses", "body_velocities")
    )
    if trigger == "pose":
        getattr(obj, f"write_{top}_link_pose_to_sim_index")(**{pose_kwarg: _payload(shape, (7,))})
    elif trigger == "velocity":
        # Zero angular velocity makes link and COM velocities equal.
        velocity = torch.zeros(shape + (6,))
        velocity[..., :3] = _payload(shape, (3,))
        getattr(obj, f"write_{top}_com_velocity_to_sim_index")(**{velocity_kwarg: velocity})
        body_velocity_expected = velocity if asset == "collection" else velocity.unsqueeze(1)
        torch.testing.assert_close(data.body_link_vel_w.torch, body_velocity_expected)
        assert data.body_link_vel_w is body_velocity
    else:
        layout = (3,) if backend == "newton" else (7,)
        coms = _payload((_NUM_INSTANCES, num_bodies), layout)
        if backend == "newton":
            from isaaclab_newton.physics import NewtonManager

            monkeypatch.setattr(NewtonManager, "add_model_change", MagicMock())
        getattr(obj, f"set_{trigger}")(coms=coms)

    for name, buffer in buffers.items():
        assert buffer.timestamp < data._sim_timestamp, name


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

_POSE, _VELOCITY = ((7,), wp.transformf), ((6,), wp.spatial_vectorf)
# written quantity -> (read-back getter with ``{top}`` as ``root`` or ``body``, (per-element shape, warp dtype))
_WRITERS = {
    "link_pose": ("{top}_link_pose_w", _POSE),
    "com_pose": ("{top}_com_pose_w", _POSE),
    "link_velocity": ("{top}_link_vel_w", _VELOCITY),
    "com_velocity": ("{top}_com_vel_w", _VELOCITY),
    "masses": ("body_mass", ((), wp.float32)),
    "inertias": ("body_inertia", ((9,), wp.float32)),
    "coms": ("body_com_pose_b", _POSE),
}


def _writer(obj, asset: str, quantity: str, selection: str):
    """Return the index or mask writer of a quantity and its payload keyword."""
    if quantity in ("masses", "inertias", "coms"):
        return getattr(obj, f"set_{quantity}_{selection}"), quantity
    top = "root" if asset == "object" else "body"
    kind = "pose" if quantity.endswith("pose") else "velocity"
    kwarg = f"root_{kind}" if asset == "object" else {"pose": "body_poses", "velocity": "body_velocities"}[kind]
    return getattr(obj, f"write_{top}_{quantity}_to_sim_{selection}"), kwarg


@pytest.mark.parametrize("selection", ["index", "mask"])
@pytest.mark.parametrize("quantity", _WRITERS)
@pytest.mark.parametrize("asset", _ASSETS)
def test_writer_reads_back_full_and_partial_writes(monkeypatch, backend, device, asset, quantity, selection):
    obj, _, top, num_bodies = _create(asset, backend, monkeypatch, device)
    obj.data.update(dt=0.01)
    getter, (trailing, wp_dtype) = _WRITERS[quantity]
    if quantity == "coms" and backend == "newton":
        # Newton stores the center of mass as a position only.
        getter, (trailing, wp_dtype) = "body_com_pos_b", ((3,), wp.vec3f)
    getter = getter.format(top=top)
    method, kwarg = _writer(obj, asset, quantity, selection)
    per_body = asset == "collection" or quantity in ("masses", "inertias", "coms")
    shape = (_NUM_INSTANCES, num_bodies) if per_body else (_NUM_INSTANCES,)

    # A torch write of every element reads back through the matching getter.
    full = _payload(shape, trailing, device=device)
    method(**{kwarg: full})
    torch.testing.assert_close(getattr(obj.data, getter).torch, full, atol=1e-5, rtol=1e-5)

    # A warp write selecting the last environment (and last body) leaves every other element untouched.
    update = _payload(shape, trailing, offset=100.0, device=device)
    selected, selectors = _select_last(update, selection, per_body)
    method(**{kwarg: wp.from_torch(selected.contiguous(), dtype=wp_dtype)}, **selectors)
    expected = full.clone()
    last = (-1, -1) if per_body else -1
    expected[last] = update[last]
    torch.testing.assert_close(getattr(obj.data, getter).torch, expected, atol=1e-5, rtol=1e-5)

    # Data with an extra environment is rejected for torch and warp inputs.
    bad = _payload((_NUM_INSTANCES + 1, *shape[1:]), trailing, device=device)
    for value in (bad, wp.from_torch(bad.contiguous(), dtype=wp_dtype)):
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: value})


@pytest.mark.parametrize("asset", _ASSETS)
def test_pose_and_velocity_writer_aliases_forward_to_canonical_writers(monkeypatch, backend, asset):
    obj, _, top, num_bodies = _create(asset, backend, monkeypatch)
    shape = (_NUM_INSTANCES,) if asset == "object" else (_NUM_INSTANCES, num_bodies)
    for quantity, getter, trailing in (("pose", f"{top}_link_pose_w", (7,)), ("velocity", f"{top}_com_vel_w", (6,))):
        for offset, selection in enumerate(("index", "mask")):
            values = _payload(shape, trailing, offset=10.0 * offset)
            method, kwarg = _writer(obj, asset, quantity, selection)
            method(**{kwarg: values})
            torch.testing.assert_close(getattr(obj.data, getter).torch, values, atol=1e-5, rtol=1e-5)


# backend-stored quantity -> (written quantity, per-element shape)
_PARTIAL_WRITES = {"pose": ("link_pose", (7,)), "velocity": ("com_velocity", (6,)), "mass": ("masses", ())}


def _read_backend(backend: str, asset: str, raw, quantity: str, num_bodies: int) -> torch.Tensor:
    """Read one quantity from backend storage as an ``(env, body, -1)`` tensor."""
    if backend == "physx":
        values = {"pose": raw.get_transforms, "velocity": raw.get_velocities, "mass": raw.get_masses}[quantity]()
        values = wp.to_torch(values)
        if asset == "collection":
            # PhysX collection views are body-major.
            values = values.reshape(num_bodies, _NUM_INSTANCES, -1).transpose(0, 1)
    elif backend == "newton":
        if quantity == "mass":
            values = wp.to_torch(raw.get_attribute("body_mass", None))
        else:
            values = wp.to_torch({"pose": raw.get_root_transforms, "velocity": raw.get_root_velocities}[quantity](None))
    else:
        from isaaclab_ov import tensor_types as TT

        bindings = {
            "object": {"pose": TT.RIGID_BODY_POSE, "velocity": TT.RIGID_BODY_VELOCITY, "mass": TT.RIGID_BODY_MASS},
            "collection": {"pose": TT.LINK_POSE, "velocity": TT.LINK_VELOCITY, "mass": TT.BODY_MASS},
        }
        values = torch.as_tensor(raw.bindings[bindings[asset][quantity]]._data)
    return values.reshape(_NUM_INSTANCES, num_bodies, -1).cpu().clone()


@pytest.mark.parametrize("selection", ["index", "mask"])
@pytest.mark.parametrize("quantity", _PARTIAL_WRITES)
@pytest.mark.parametrize("asset", _ASSETS)
def test_partial_write_preserves_unselected_backend_elements(monkeypatch, request, backend, asset, quantity, selection):
    if backend == "ovphysx" and asset == "collection" and quantity != "mass":
        # Product bug: OVPhysX pushes whole environment rows from a pose/velocity buffer that a partial write does
        # not refresh first, so the unselected bodies of the selected environment receive stale (here zero) state.
        request.applymarker(
            pytest.mark.xfail(
                raises=AssertionError,
                strict=True,
                reason="OVPhysX partial body writes overwrite unselected bodies of the selected environment",
            )
        )
    obj, raw, _, num_bodies = _create(asset, backend, monkeypatch)
    written, trailing = _PARTIAL_WRITES[quantity]
    per_body = asset == "collection" or quantity == "mass"
    shape = (_NUM_INSTANCES, num_bodies) if per_body else (_NUM_INSTANCES,)
    values = _payload(shape, trailing, offset=100.0)
    before = _read_backend(backend, asset, raw, quantity, num_bodies)

    selected, selectors = _select_last(values, selection, per_body)
    method, kwarg = _writer(obj, asset, written, selection)
    method(**{kwarg: selected}, **selectors)

    expected = before.clone()
    expected[-1, -1] = values.reshape(_NUM_INSTANCES, num_bodies, -1)[-1, -1]
    torch.testing.assert_close(_read_backend(backend, asset, raw, quantity, num_bodies), expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Family-specific behavior
# ---------------------------------------------------------------------------


def test_rigid_object_external_wrench_frames(monkeypatch, backend):
    """Forward local and world wrenches through the real writer in each backend's frame."""
    obj, raw, _, _ = _create("object", backend, monkeypatch)
    # A known 90-degree rotation about Z keeps the local-frame expectation independent of production.
    root_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5], [4.0, 5.0, 6.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5]]
    )
    obj.write_root_link_pose_to_sim_index(root_pose=root_pose)
    if backend == "newton":
        # Seed the body pose the native forward kinematics would publish.
        obj.data._sim_bind_body_link_pose_w.assign(wp.from_torch(root_pose, dtype=wp.transformf))
    if backend == "physx":
        raw.apply_forces_and_torques_at_position = MagicMock()
    composer = obj.permanent_wrench_composer
    forces = torch.arange(1.0, 7.0).reshape(2, 1, 3)
    torques = forces + 10.0

    for is_global in (False, True):
        composer.reset()
        composer.set_forces_and_torques_index(forces=forces, torques=torques, is_global=is_global)
        with patch.object(composer, "compose_to_body_frame", wraps=composer.compose_to_body_frame) as compose:
            obj.write_data_to_sim()
        assert compose.call_count == int(is_global and backend == "newton")

        expected_force, expected_torque = forces, torques
        if backend == "physx":
            call = raw.apply_forces_and_torques_at_position.call_args.kwargs
            assert call["is_global"] is is_global
            assert call["position_data"] is None
            actual_force, actual_torque = call["force_data"].numpy(), call["torque_data"].numpy()
        else:
            if not is_global:
                # A 90-degree Z rotation maps body (x, y, z) to world (-y, x, z).
                expected_force = torch.stack((-forces[..., 1], forces[..., 0], forces[..., 2]), dim=-1)
                expected_torque = torch.stack((-torques[..., 1], torques[..., 0], torques[..., 2]), dim=-1)
            if backend == "newton":
                packed = obj.data._sim_bind_body_external_wrench.numpy()
            else:
                from isaaclab_ov import tensor_types as TT

                packed = raw.bindings[TT.RIGID_BODY_WRENCH]._data.reshape(2, 1, 9)
                np.testing.assert_allclose(packed[..., 6:9], root_pose[:, None, :3].numpy())
            actual_force, actual_torque = packed[..., :3], packed[..., 3:6]
        np.testing.assert_allclose(actual_force.reshape(2, 1, 3), expected_force.numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(actual_torque.reshape(2, 1, 3), expected_torque.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collection_reshape_data_to_view_3d(monkeypatch, backend, device, dtype):
    """Environment-major data becomes body-major view data, keeping the input dtype and array type."""
    obj, _, _, num_bodies = _create("collection", backend, monkeypatch, device)
    data = torch.arange(_NUM_INSTANCES * num_bodies * 4, dtype=dtype, device=device).reshape(
        _NUM_INSTANCES, num_bodies, 4
    )
    expected = data.permute(1, 0, 2).reshape(num_bodies * _NUM_INSTANCES, 4)

    view = obj.reshape_data_to_view_3d(data, 4, device=device)
    assert isinstance(view, torch.Tensor) and view.is_contiguous()
    assert (view.dtype, view.device) == (dtype, data.device)
    torch.testing.assert_close(view, expected)

    if dtype == torch.float32:
        warp_view = obj.reshape_data_to_view_3d(wp.from_torch(data, dtype=wp.float32), 4, device=device)
        assert isinstance(warp_view, wp.array) and warp_view.dtype == wp.float32 and str(warp_view.device) == device
        torch.testing.assert_close(wp.to_torch(warp_view), expected)
    if device != "cpu":
        # Torch inputs move to the requested device.
        torch.testing.assert_close(obj.reshape_data_to_view_3d(data, 4, device="cpu"), expected.cpu())
