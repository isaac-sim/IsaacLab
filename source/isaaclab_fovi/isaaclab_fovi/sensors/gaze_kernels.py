# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels of the virtual gaze, one launch each per camera pose update and published output.

:func:`project_direction` follows :meth:`fovi.sensing.projection.CameraModel.project` operation for operation, in
float32. Camera-frame directions follow the ROS (OpenCV) convention: x right, y down, z forward.
"""

from __future__ import annotations

import math

import torch
import warp as wp
from fovi.sensing.projection import CameraModel

vec8f = wp.types.vector(length=8, dtype=wp.float32)

FLOAT32_EPS = float(torch.finfo(torch.float32).eps)
DEG_TO_RAD = math.pi / 180.0


_WARP_STREAMS: dict[tuple[int, int], wp.Stream] = {}
"""Warp streams by the CUDA device index and stream handle of the PyTorch stream they launch on; default streams of
different devices share a handle."""


def warp_stream(device: torch.device | str) -> wp.Stream | None:
    """The Warp stream of PyTorch's current stream on a CUDA device, or None on the CPU."""
    device = torch.device(device)
    if device.type != "cuda":
        wp.init()
        return None
    torch_current = torch.cuda.current_stream(device)
    key = (torch_current.device_index, torch_current.cuda_stream)
    stream = _WARP_STREAMS.get(key)
    if stream is None:
        wp.init()
        stream = wp.get_stream(wp.device_from_torch(device))
        # A temporary alias of a live Warp-owned stream would unregister it when destroyed; cached aliases persist.
        if stream.cuda_stream != key[1]:
            stream = wp.stream_from_torch(torch_current)
        _WARP_STREAMS[key] = stream
    return stream


def torch_stream(device: torch.device | str) -> wp.ScopedStream | wp.ScopedDevice:
    """Scope that launches Warp kernels on the tensors' device and, on CUDA, PyTorch's current stream."""
    stream = warp_stream(device)
    if stream is None:
        return wp.ScopedDevice(wp.device_from_torch(torch.device(device)))
    return wp.ScopedStream(stream, sync_enter=False, sync_exit=False)


def record_launch(
    kernel, dim, inputs, outputs=(), device: torch.device | str = "cpu", block_dim: int = 256
) -> wp.Launch:
    """A launch recorded once and replayed each step with :meth:`warp.Launch.launch`, skipping argument packing."""
    wp.init()
    return wp.launch(
        kernel, dim=dim, inputs=inputs, outputs=outputs, device=str(device), block_dim=block_dim, record_cmd=True
    )


@wp.struct
class CameraParams:
    """A :class:`~fovi.sensing.projection.CameraModel` for Warp kernels."""

    fisheye: wp.int32
    focal: wp.vec2f
    center: wp.vec2f
    size: wp.vec2f
    """Image width and height [px]."""
    distortion: vec8f
    """Pinhole (k1, k2, p1, p2, k3, k4, k5, k6) or fisheye (k1, k2, k3, k4), zero-padded."""
    has_circle: wp.int32
    circle: wp.vec3f
    max_angle: wp.float32
    """Largest angle from the optical axis [rad]."""


def camera_params(model: CameraModel) -> CameraParams:
    """The Warp form of a camera model."""
    params = CameraParams()
    params.fisheye = int(model.model == "fisheye")
    fx, fy, cx, cy = model.intrinsics
    params.focal = wp.vec2f(fx, fy)
    params.center = wp.vec2f(cx, cy)
    height, width = model.image_size
    params.size = wp.vec2f(width, height)
    params.distortion = vec8f(*model.distortion, *([0.0] * (8 - len(model.distortion))))
    params.has_circle = int(model.image_circle is not None)
    params.circle = wp.vec3f(*(model.image_circle or (0.0, 0.0, 1.0)))
    # fovi compares float32 angles against the limit converted to float32.
    params.max_angle = float(torch.tensor(math.radians(model.max_angle_deg), dtype=torch.float32))
    return params


@wp.func
def project_direction(d: wp.vec3f, camera: CameraParams) -> wp.vec3f:
    """Pixel (x, y) of a camera-frame direction, and 1 where the camera sees it, else 0."""
    eps = wp.float32(FLOAT32_EPS)
    z = d[2]
    lateral = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    theta = wp.atan2(lateral, z)
    k = camera.distortion
    x = wp.float32(0.0)
    y = wp.float32(0.0)
    if camera.fisheye == 0:
        depth = wp.max(z, eps)
        u = d[0] / depth
        v = d[1] / depth
        r2 = u * u + v * v
        radial = (1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[4]))) / (1.0 + r2 * (k[5] + r2 * (k[6] + r2 * k[7])))
        x = u * radial + 2.0 * k[2] * u * v + k[3] * (r2 + 2.0 * u * u)
        y = v * radial + k[2] * (r2 + 2.0 * v * v) + 2.0 * k[3] * u * v
    else:
        t2 = theta * theta
        radius = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
        scale = 1.0 / wp.max(z, eps)
        if lateral > eps:
            scale = radius / wp.max(lateral, eps)
        x = d[0] * scale
        y = d[1] * scale
    px = x * camera.focal[0] + camera.center[0]
    py = y * camera.focal[1] + camera.center[1]
    valid = wp.isfinite(px) and wp.isfinite(py)
    valid = valid and px >= -0.5 and px < camera.size[0] - 0.5 and py >= -0.5 and py < camera.size[1] - 0.5
    if camera.has_circle != 0:
        dx = px - camera.circle[0]
        dy = py - camera.circle[1]
        valid = valid and dx * dx + dy * dy <= camera.circle[2] * camera.circle[2]
    valid = valid and theta <= camera.max_angle
    if camera.fisheye == 0:
        valid = valid and z > 0.0
    return wp.vec3f(px, py, wp.where(valid, 1.0, 0.0))


@wp.func
def gaze_rotation(angles_deg: wp.vec2f) -> wp.mat33f:
    """Pan-then-tilt rotation of a gaze (tilt, pan) [deg], as :func:`fovi.sensing.projection.gaze_rotation`."""
    tilt = angles_deg[0] * wp.float32(DEG_TO_RAD)
    pan = angles_deg[1] * wp.float32(DEG_TO_RAD)
    x = -wp.cos(tilt) * wp.sin(pan)
    y = -wp.sin(tilt)
    z = wp.cos(tilt) * wp.cos(pan)
    pitch = wp.atan2(x, z)
    roll = wp.atan2(-y, wp.sqrt(x * x + z * z))
    cp = wp.cos(pitch)
    sp = wp.sin(pitch)
    cr = wp.cos(roll)
    sr = wp.sin(roll)
    return wp.mat33f(cp, sp * sr, sp * cr, 0.0, cr, -sr, -sp, cp * sr, cp * cr)


@wp.func
def gaze_pose(
    orientation: wp.quatf, gaze: wp.mat33f, basis: wp.mat33f, to_opengl: wp.quatf, to_world: wp.quatf
) -> wp.quatf:
    """The world orientation (world convention) of a camera at ``orientation`` turned by a gaze rotation."""
    relative = basis * gaze * wp.transpose(basis)
    return wp.normalize(orientation * (to_opengl * wp.quat_from_matrix(relative) * to_world))


@wp.kernel(enable_backward=False)
def set_gaze_kernel(
    angles_deg: wp.array(dtype=wp.vec2f),
    stored_angles: wp.array(dtype=wp.vec2f),
    rotation: wp.array(dtype=wp.mat33f),
):
    """Store each environment's gaze angles and their gaze rotation."""
    i = wp.tid()
    angles = angles_deg[i]
    stored_angles[i] = angles
    rotation[i] = gaze_rotation(angles)


@wp.kernel(enable_backward=False)
def apply_gaze_kernel(
    angles_deg: wp.array(dtype=wp.vec2f),
    basis: wp.mat33f,
    to_opengl: wp.quatf,
    to_world: wp.quatf,
    mask: wp.array(dtype=wp.bool),
    use_mask: wp.int32,
    orientation: wp.array(dtype=wp.quatf),
    rotation: wp.array(dtype=wp.mat33f),
):
    """Turn each updated camera and save the rotation used for its capture."""
    i = wp.tid()
    if use_mask == 0 or mask[i]:
        gaze = gaze_rotation(angles_deg[i])
        rotation[i] = gaze
        orientation[i] = gaze_pose(orientation[i], gaze, basis, to_opengl, to_world)


@wp.kernel(enable_backward=False)
def apply_gaze_indexed_kernel(
    angles_deg: wp.array(dtype=wp.vec2f),
    basis: wp.mat33f,
    to_opengl: wp.quatf,
    to_world: wp.quatf,
    indices: wp.array(dtype=wp.int32),
    orientation: wp.array(dtype=wp.quatf),
    rotation: wp.array(dtype=wp.mat33f),
):
    """Turn the indexed cameras' world orientations by their gaze, and record their gaze rotations."""
    i = indices[wp.tid()]
    gaze = gaze_rotation(angles_deg[i])
    rotation[i] = gaze
    orientation[i] = gaze_pose(orientation[i], gaze, basis, to_opengl, to_world)


@wp.func
def sees(ray: wp.vec3f, gaze: wp.mat33f, frame: wp.mat33f, camera: CameraParams) -> wp.bool:
    """Whether the physical camera sees a rendered ray (ROS frame) turned by a gaze; ``frame`` is the mount roll."""
    return project_direction(frame * gaze * wp.transpose(frame) * ray, camera)[2] > 0.5


@wp.kernel(enable_backward=False)
def publish_grid_kernel(
    source: wp.array3d(dtype=wp.uint8),
    indices: wp.array(dtype=wp.int32),
    footprint: wp.array(dtype=wp.bool),
    updated: wp.array(dtype=wp.bool),
    rays: wp.array(dtype=wp.vec3f),
    rotation: wp.array(dtype=wp.mat33f),
    frame: wp.mat33f,
    camera: CameraParams,
    mask_to_sensor: wp.int32,
    target: wp.array3d(dtype=wp.uint8),
):
    """Copy the updated environments' rendered samples into the upright grid, zeroing samples outside the footprint
    or, when masking, the physical camera's view.

    ``source`` and ``target`` hold each sample's channels as raw bytes, so one kernel serves every output type.
    """
    env, cell = wp.tid()
    if not updated[env]:
        return
    sample = indices[cell]
    keep = footprint[sample]
    if keep and mask_to_sensor != 0:
        keep = sees(rays[sample], rotation[env], frame, camera)
    for k in range(target.shape[2]):
        if keep:
            target[env, cell, k] = source[env, sample, k]
        else:
            target[env, cell, k] = wp.uint8(0)
