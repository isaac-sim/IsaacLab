# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent numerical contracts for device-callable math."""

import math

import numpy as np
import pytest
import warp as wp

from isaaclab.utils.warp import math as math_kernels
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


def test_heading_and_inverse_rotation_use_body_and_world_frames():
    with wp.ScopedDevice("cpu"):
        s = 2.0**-0.5
        quat = wp.array([[0, 0, s, s], [0, 0, -s, s]], dtype=wp.quatf)
        vector = wp.array([[1, 2, 3], [4, 5, 6]], dtype=wp.vec3f)
        forward = wp.array([[1, 0, 0], [0, 1, 0]], dtype=wp.vec3f)
        rotated = wp.empty(2, dtype=wp.vec3f)
        heading = wp.empty(2, dtype=float)
        for inverse, yaw, shape in [
            (math_kernels.quat_rotate_inv_kernel[1], math_kernels.heading_kernel[1], (2,)),
            (math_kernels.quat_rotate_inv_kernel[2], math_kernels.heading_kernel[2], (1, 2)),
        ]:
            wp.launch(inverse, shape, [quat.reshape(shape), vector.reshape(shape), rotated.reshape(shape)])
            wp.launch(yaw, shape, [forward.reshape(shape), quat.reshape(shape), heading.reshape(shape)])
            np.testing.assert_allclose(rotated.numpy(), [[2, -1, 3], [-5, 4, 6]], atol=1e-6)
            np.testing.assert_allclose(heading.numpy(), [math.pi / 2, 0], atol=1e-6)


def test_interval_scaling_preserves_midpoints_without_clamping():
    with wp.ScopedDevice("cpu"):
        bounds = wp.array([[[-4, 8], [2, 6]]], dtype=wp.vec2f)
        result = wp.empty_like(bounds)
        for factor, expected in [(0.5, [[[-1, 5], [3, 5]]]), (-1.0, [[[8, -4], [6, 2]]])]:
            wp.launch(math_kernels.scale_interval_kernel, (1, 2), [bounds, factor, result])
            np.testing.assert_array_equal(result.numpy(), expected)


def test_jacobian_shift_preserves_angular_rows_and_maps_omitted_root():
    with wp.ScopedDevice("cpu"):
        s = 2.0**-0.5
        pose = wp.array([[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, s, s]]], dtype=wp.transformf)
        com = wp.array([[[1, 0, 0], [2, 0, 0]]], dtype=wp.vec3f)
        jacobian = wp.array([[[[1, -1], [2, 4], [3, 5], [0, 1], [0, 0], [2, 0]]]], dtype=float)
        result = wp.empty_like(jacobian)
        for offset, expected_linear in [(0, [[1, -1], [0, 4], [3, 5]]), (1, [[5, -1], [2, 4], [3, 3]])]:
            wp.launch(math_kernels.shift_jacobian_reference_kernel, (1, 1, 2), [pose, com, offset, jacobian, result])
            np.testing.assert_allclose(result.numpy()[0, 0, :3], expected_linear, atol=1e-6)
            np.testing.assert_array_equal(result.numpy()[0, 0, 3:], [[0, 1], [0, 0], [2, 0]])


def test_shared_math_ownership(source_checkout_root):
    """Shared equations have one solver-independent owner; factories never run per step."""
    import ast

    from isaaclab.utils.warp import math

    for operation in (math.quat_rotate_inv_kernel, math.heading_kernel):
        assert all(kernel.module.name == math.__name__ for kernel in operation.values())
    source = source_checkout_root / "source"
    core = source / "isaaclab/isaaclab"
    for name in ("math", "kernel_factory"):
        owner = core / f"utils/warp/{name}.py"
        assert owner.is_file()
        for node in ast.walk(ast.parse(owner.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name.lstrip("_").startswith("make_"):
                assert not node.name.endswith("_kernel"), node.name
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
                        and getattr(call.func, "id", getattr(call.func, "attr", "")) == "make_binary"
                        for call in ast.walk(node)
                    ), file
