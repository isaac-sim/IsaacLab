# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent numerical contracts for device-callable math."""

import math

import numpy as np
import pytest
import warp as wp

from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import math as warp_math
from isaaclab.utils.warp.math import (
    acceleration_at_offset,
    body_ang_vel_from_root,
    body_lin_vel_from_root,
    quat_error_magnitude,
    rotate_vec_to_body_frame,
    velocity_at_offset,
)

pytestmark = pytest.mark.unit


@wp.kernel
def _angles(a: wp.array(dtype=wp.quatf), b: wp.array(dtype=wp.quatf), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = quat_error_magnitude(a[i], b[i])


def test_quaternion_error_uses_xyzw_and_shortest_rotation():
    s = 2.0**-0.5
    identity = [0, 0, 0, 1]
    a = [identity, identity, identity, identity, [s, 0, 0, s], [0, 0, math.cos(0.05), math.sin(0.05)]]
    b = [identity, [0, 0, 0, -1], [s, 0, 0, s], [1, 0, 0, 0], [0, s, 0, s], [0, 0, -math.cos(0.05), math.sin(0.05)]]
    expected = [0, 0, math.pi / 2, math.pi, 2 * math.pi / 3, 0.2]
    # Scale invariance is a Warp contract; Torch parity is only claimed for unit inputs.
    a.append([0, 0, 0, 2])
    b.append([-3 * s, 0, 0, -3 * s])
    expected.append(math.pi / 2)
    a.append(identity)
    b.append([float("nan"), 0, 0, 1])
    expected.append(float("nan"))
    out = wp.empty(len(a), dtype=float, device="cpu")
    wp.launch(
        _angles,
        len(a),
        [wp.array(a, dtype=wp.quatf, device="cpu"), wp.array(b, dtype=wp.quatf, device="cpu"), out],
        device="cpu",
    )
    np.testing.assert_allclose(out.numpy(), expected, atol=2e-6, equal_nan=True)


@wp.kernel
def _transport(out: wp.array(dtype=wp.vec3f)):
    velocity = wp.spatial_vector(1.0, 2.0, 3.0, 0.0, 0.0, 2.0)
    acceleration = wp.spatial_vector(4.0, 5.0, 6.0, 0.0, 0.0, 3.0)
    offset = wp.vec3f(1.0, 0.0, 0.0)
    pose = wp.transform(wp.vec3f(7.0), wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), wp.pi / 2.0))
    out[0] = velocity_at_offset(velocity, offset)
    out[1] = acceleration_at_offset(acceleration, wp.spatial_bottom(velocity), offset)
    out[2] = body_lin_vel_from_root(pose, velocity)
    out[3] = body_ang_vel_from_root(pose, velocity)
    out[4] = rotate_vec_to_body_frame(wp.vec3f(1.0, 2.0, 3.0), pose)


def test_spatial_transport_is_linear_first_and_in_the_input_frame():
    out = wp.empty(5, dtype=wp.vec3f, device="cpu")
    wp.launch(_transport, 1, [out], device="cpu")
    np.testing.assert_allclose(out.numpy(), [[1, 4, 3], [0, 8, 6], [2, -1, 3], [0, 0, 2], [2, -1, 3]], atol=2e-6)


@pytest.mark.parametrize("device", test_devices())
def test_heading_and_inverse_rotation_preserve_strides_and_replay(device):
    with wp.ScopedDevice(device):
        s = 2.0**-0.5
        quaternions = np.array([[0, 0, s, s], [0, 0, -s, s], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
        vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]], dtype=np.float32)
        forwards = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
        for shape in ((4,), (2, 2)):
            rank = len(shape)
            # Independent strides and padding expose incorrect flattening or shared-stride assumptions.
            arrays, stores, selections = [], [], []
            for step, dtype, values in (
                (2, wp.quatf, quaternions),
                (3, wp.vec3f, vectors),
                (4, wp.vec3f, forwards),
                (5, wp.vec3f, None),
                (6, wp.float32, None),
            ):
                store = wp.full((*shape[:-1], shape[-1] * step), dtype(99.0))
                selection = (slice(None),) * (rank - 1) + (slice(None, None, step),)
                array = store[selection]
                if values is not None:
                    array.assign(values.reshape(*shape, -1))
                arrays.append(array)
                stores.append(store)
                selections.append(selection)
            quat, vector, forward, rotated, heading = arrays
            inverse, yaw = warp_math.quat_rotate_inv[rank], warp_math.heading[rank]
            wp.launch(inverse, shape, [quat, vector, rotated])
            wp.launch(yaw, shape, [forward, quat, heading])
            np.testing.assert_allclose(
                rotated.numpy().reshape(4, 3), [[2, -1, 3], [-5, 4, 6], [7, 8, 9], [1, 2, -3]], atol=1e-6
            )
            np.testing.assert_allclose(
                heading.numpy().reshape(4), [math.pi / 2, 0, -math.pi / 2, -math.pi / 2], atol=1e-6
            )
            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    wp.launch(inverse, shape, [quat, vector, rotated])
                    wp.launch(yaw, shape, [forward, quat, heading])
                quat.fill_(wp.quatf(0, 0, 0, 1))
                vector.fill_(wp.vec3f(2, 3, 4))
                forward.fill_(wp.vec3f(0, 1, 0))
                rotated.fill_(wp.vec3f(99.0))
                heading.fill_(99.0)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(rotated.numpy(), np.tile([2, 3, 4], (*shape, 1)))
                np.testing.assert_allclose(heading.numpy(), math.pi / 2, atol=1e-6)
            for store, selection in zip(stores, selections, strict=True):
                padding = np.ones(store.shape, dtype=bool)
                padding[selection] = False
                np.testing.assert_array_equal(store.numpy()[padding], 99.0)


def test_interval_scaling_preserves_midpoints_without_clamping():
    with wp.ScopedDevice("cpu"):
        bounds = wp.array([[[-4, 8], [2, 6]]], dtype=wp.vec2f)
        result = wp.empty_like(bounds)
        for factor, expected in [(0.5, [[[-1, 5], [3, 5]]]), (-1.0, [[[8, -4], [6, 2]]])]:
            wp.launch(warp_math.scale_intervals, (1, 2), [bounds, factor, result])
            np.testing.assert_array_equal(result.numpy(), expected)


def test_jacobian_shift_preserves_angular_rows_and_maps_omitted_root():
    with wp.ScopedDevice("cpu"):
        s = 2.0**-0.5
        pose = wp.array([[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, s, s]]], dtype=wp.transformf)
        com = wp.array([[[1, 0, 0], [2, 0, 0]]], dtype=wp.vec3f)
        jacobian = wp.array([[[[1, -1], [2, 4], [3, 5], [0, 1], [0, 0], [2, 0]]]], dtype=float)
        result = wp.empty_like(jacobian)
        for offset, expected_linear in [(0, [[1, -1], [0, 4], [3, 5]]), (1, [[5, -1], [2, 4], [3, 3]])]:
            wp.launch(warp_math.shift_jacobian_reference, (1, 1, 2), [pose, com, offset, jacobian, result])
            np.testing.assert_allclose(result.numpy()[0, 0, :3], expected_linear, atol=1e-6)
            np.testing.assert_array_equal(result.numpy()[0, 0, 3:], [[0, 1], [0, 0], [2, 0]])


def test_shared_math_ownership(source_checkout_root):
    """Core owns equations; Warp owns elementwise generation and operation-specific compilation."""
    import ast

    from isaaclab.utils.warp import math

    assert not any(name.endswith(("_kernel", "_kernels")) for name in vars(math)), "Name shared math by operation"
    mapped = [*math.quat_rotate_inv.values(), *math.heading.values(), math.scale_intervals]
    assert all(kernel.module is not wp.get_module(math.__name__) for kernel in mapped)
    source = source_checkout_root / "source"
    core = source / "isaaclab/isaaclab"
    assert not (core / "utils/warp/kernel_factory.py").exists(), "Use Warp's elementwise generator"
    owner = core / "utils/warp/math.py"
    for node in ast.walk(ast.parse(owner.read_text())):
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        assert not any(
            m.startswith(
                (
                    "isaaclab.assets",
                    "isaaclab.sensors",
                    "isaaclab_newton",
                    "isaaclab_physx",
                    "isaaclab_ov",
                    "newton",
                    "isaacsim",
                )
            )
            for m in modules
        ), owner
    retired = {
        "compute_heading_w_func",
        "quat_apply_inverse_1D_kernel",
        "quat_apply_inverse_2D_kernel",
        "root_heading_w",
        "body_heading_w",
        "compute_soft_joint_pos_limits_func",
        "update_soft_joint_pos_limits",
        "shift_jacobian_com_to_origin",
    }
    for backend in ("ov", "physx", "newton"):
        for relative in ("assets/kernels.py", "assets/articulation/kernels.py"):
            file = source / f"isaaclab_{backend}/isaaclab_{backend}" / relative
            assert not retired & {n.name for n in ast.parse(file.read_text()).body if isinstance(n, ast.FunctionDef)}
    assert not (source / "isaaclab_newton/isaaclab_newton/kernels/state_kernels.py").exists()
    for relative, forbidden in (
        ("isaaclab/isaaclab/sensors/ray_caster/kernels.py", "quat_yaw_only"),
        (
            "isaaclab_tasks_experimental/isaaclab_tasks_experimental/core/reorient/reorient_warp_env.py",
            "rotation_distance",
        ),
    ):
        assert forbidden not in {
            n.name for n in ast.parse((source / relative).read_text()).body if isinstance(n, ast.FunctionDef)
        }
    for package in source.iterdir():
        for file in (package / package.name).rglob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "isaaclab_newton.kernels.state_kernels", file
                if isinstance(node, ast.Import):
                    assert all(a.name != "isaaclab_newton.kernels.state_kernels" for a in node.names), file
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    assert not any(
                        isinstance(call, ast.Call)
                        and getattr(call.func, "id", getattr(call.func, "attr", "")).lstrip("_") == "make_binary"
                        for call in ast.walk(node)
                    ), file
