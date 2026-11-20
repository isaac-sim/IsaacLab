# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gaze action, its observation and its teacher, without a simulator."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.mdp.gaze import (
    GazeAction,
    gaze_reset_kernel,
    gaze_state,
    gaze_step_kernel,
    teacher_labels_kernel,
)
from isaaclab_fovi.mdp.gaze_cfg import GazeActionCfg
from isaaclab_fovi.sensors.gaze_kernels import camera_params, torch_stream

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless

DEVICES = test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA)


def angles_to_direction(angles_deg: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit directions of gaze angles [deg], shape (..., 2) to (..., 3): up and left are positive."""
    tilt, pan = angles_deg.deg2rad().unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def direction_to_angles(direction: torch.Tensor) -> torch.Tensor:
    """Gaze angles [deg] of camera-frame directions, shape (..., 3) to (..., 2): the reference for the kernels."""
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1).rad2deg()


def clamp_gaze(gaze: torch.Tensor, limit: float) -> torch.Tensor:
    """Gazes scaled back onto the circle of radius ``limit`` when they leave it: the reference for the step kernel."""
    return gaze * (limit / gaze.norm(dim=-1, keepdim=True).clamp_min(limit))


def test_gaze_step_kernel_matches_the_torch_update():
    generator = torch.Generator().manual_seed(4)
    count, step, limit, dt, radius = 2048, 0.1, 1.0, 1 / 60, 23.6
    actions = torch.randn(count, 2, generator=generator) * 1.5
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.75
    expected = clamp_gaze(gaze + actions.clamp(-1, 1) * step, limit)
    expected_velocity = (expected - gaze) / dt

    # The step starts from the camera's stored angles, not from the state it records.
    state = torch.randn(count, 4, generator=generator)
    angles = gaze * radius
    with torch_stream("cpu"):
        wp.launch(
            gaze_step_kernel,
            dim=count,
            inputs=[
                wp.from_torch(actions, dtype=wp.vec2f),
                wp.from_torch(state, dtype=wp.vec4f),
                wp.from_torch(angles, dtype=wp.vec2f),
                step,
                limit,
                dt,
                radius,
            ],
        )
    torch.testing.assert_close(state[:, :2], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state[:, 2:], expected_velocity, atol=2e-3, rtol=1e-5)
    torch.testing.assert_close(angles, expected * radius, atol=1e-5, rtol=1e-6)


def mount_frame(rotation_deg: float) -> torch.Tensor:
    """The camera's ``_mask_frame`` for an image rolled by ``rotation_deg``: upright gaze frame to physical frame."""
    angle = torch.tensor(rotation_deg).deg2rad()
    c, s = float(angle.cos()), float(angle.sin())
    return torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


# The kernel's block size depends on the device; the image roll is rotated across the two.
TEACHER_CASES = [("cpu", 0.0)] + [(device, 35.0) for device in DEVICES if device != "cpu"]


@pytest.mark.parametrize(("device", "rotation_deg"), TEACHER_CASES)
def test_teacher_labels_kernel_matches_the_torch_labels(device, rotation_deg):
    camera = CameraModel("pinhole", (480, 640), (400.0, 400.0, 319.5, 239.5), distortion=(-0.1, 0.02, 0.0, 0.0))
    field_radius, step, limit = 30.0, 0.1, 1.0
    frame = mount_frame(rotation_deg)
    generator = torch.Generator().manual_seed(5)
    count = 4099
    # Each camera has its own mount, as on a moving head.
    mount = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=-1)
    positions = torch.randn(count, 3, generator=generator)
    lateral = torch.randn(count, 2, generator=generator) * 0.6
    local = torch.cat((lateral, torch.rand(count, 1, generator=generator) + 0.2), -1)
    targets = positions + quat_apply(mount, local)
    orientations = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=-1)
    offset = torch.tensor([0.0, 0.0, 0.3])
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.7

    # The torch path the kernel replaces: the point fixed to each target, the teacher, then the labels and logs.
    points = targets + quat_apply(orientations, offset.expand(count, -1))
    direction = torch.nn.functional.normalize(quat_apply_inverse(mount, points - positions), dim=-1)

    def teachable(direction: torch.Tensor) -> torch.Tensor:
        # Gazes are measured in the upright image frame, which the frame carries into the physical camera's.
        desired = direction_to_angles(direction @ frame) / field_radius
        return camera.project(direction)[1] & (desired.norm(dim=-1) <= limit)

    upright = direction @ frame
    desired = direction_to_angles(upright) / field_radius
    valid = teachable(direction)
    # A target within 1e-4 rad of the image edge, the lens's widest angle or the gaze limit may land on either side
    # in float32; every other target must agree exactly.
    nudges = 1e-4 * torch.cat((torch.eye(3), -torch.eye(3)))
    decisive = torch.stack(
        [teachable(torch.nn.functional.normalize(direction + nudge, dim=-1)) == valid for nudge in nudges]
    ).all(0)
    action = torch.where(valid[:, None], ((desired - gaze) / step).clamp(-1, 1), 0.0)
    looking = angles_to_direction(gaze * field_radius)
    error = torch.atan2(torch.cross(looking, upright, dim=-1).norm(dim=-1), (looking * upright).sum(-1)).rad2deg()
    expected = torch.cat((action, valid[:, None].float()), dim=-1)

    labels = torch.empty(count, 3, device=device)
    log = torch.full((2,), float("nan"), device=device)
    block = 1024 if torch.device(device).type == "cuda" else 1
    with torch_stream(device):
        wp.launch(
            teacher_labels_kernel,
            dim=(1, block),
            inputs=[
                wp.from_torch(targets.to(device), dtype=wp.vec3f),
                wp.from_torch(orientations.to(device), dtype=wp.quatf),
                wp.vec3f(*offset.tolist()),
                wp.from_torch(positions.to(device), dtype=wp.vec3f),
                wp.from_torch(mount.to(device), dtype=wp.quatf),
                wp.from_torch((gaze * field_radius).to(device), dtype=wp.vec2f),
                wp.mat33f(*frame.flatten().tolist()),
                field_radius,
                step,
                limit,
                camera_params(camera),
            ],
            outputs=[wp.from_torch(labels), wp.from_torch(log)],
            block_dim=block,
        )
    labels, log = labels.cpu(), log.cpu()
    assert valid[decisive].any() and (~valid[decisive]).any() and decisive.float().mean() > 0.99
    torch.testing.assert_close(labels[decisive], expected[decisive], atol=1e-4, rtol=0)
    labelled = labels[:, 2] > 0.5
    assert float(log[1]) == pytest.approx(float(labelled.float().mean()))
    assert float(log[0]) == pytest.approx(float(error[labelled].mean()), rel=1e-4)


def test_teacher_gaze_fixates_the_target_through_a_rolled_mount():
    """Following a teacher label puts the fixation, which reaches the physical camera at frame * gaze * z, on the
    target."""
    rotation_deg = 120.0
    camera = CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))
    field_radius, step = 40.0, 10.0
    frame = mount_frame(rotation_deg)
    direction = torch.nn.functional.normalize(torch.tensor([[0.25, -0.15, 1.0], [-0.3, 0.2, 1.0]]), dim=-1)
    count = len(direction)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)
    labels, log = torch.empty(count, 3), torch.empty(2)
    with torch_stream("cpu"):
        wp.launch(
            teacher_labels_kernel,
            dim=(1, 1),
            inputs=[
                wp.from_torch(direction, dtype=wp.vec3f),
                wp.from_torch(identity, dtype=wp.quatf),
                wp.vec3f(0.0, 0.0, 0.0),
                wp.zeros(count, dtype=wp.vec3f),
                wp.from_torch(identity, dtype=wp.quatf),
                wp.zeros(count, dtype=wp.vec2f),
                wp.mat33f(*frame.flatten().tolist()),
                field_radius,
                step,
                1.0,
                camera_params(camera),
            ],
            outputs=[wp.from_torch(labels), wp.from_torch(log)],
            block_dim=1,
        )
    assert (labels[:, 2] == 1).all()
    gaze = labels[:, :2] * step
    fixation = angles_to_direction(gaze * field_radius) @ frame.T
    torch.testing.assert_close(fixation, direction, atol=1e-5, rtol=0)
    # The error is the gaze's angle from the target before the step: the gaze starts on the optical axis.
    torch.testing.assert_close(log[0], direction[:, 2].acos().rad2deg().mean(), atol=1e-3, rtol=0)


def test_gaze_config_rejects_bounds_that_divide_by_zero_and_unknown_mounts():
    GazeActionCfg(asset_name="camera").validate_config()
    GazeActionCfg(asset_name="camera", mount="tracked").validate_config()
    for field, value in (
        ("step_size", 0.0),
        ("step_size", float("nan")),
        ("gaze_limit", math.inf),
        ("gaze_limit", "1.0"),
        ("mount", "head"),
    ):
        cfg = GazeActionCfg(asset_name="camera")
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=field):
            cfg.validate_config()
    # The config names its term lazily, so Isaac Lab's action manager resolves it to the gaze action.
    assert GazeActionCfg(asset_name="camera").class_type._resolve() is GazeAction


def test_process_actions_moves_the_gaze_by_the_supplied_actions():
    """The term reads whatever actions it is handed: the manager's slice in place, or another tensor."""
    num_envs = 4
    angles = ProxyArray(wp.zeros(num_envs, dtype=wp.vec2f, device="cpu"))
    set_angles = []
    camera = SimpleNamespace(gaze_angles=angles, set_gaze_angles=lambda value: set_angles.append(value))
    term = object.__new__(GazeAction)
    term._env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=0.5)
    term.cfg = GazeActionCfg(asset_name="camera", step_size=0.1, gaze_limit=1.0)
    term._asset = camera
    term.field_radius_deg = 20.0
    term._state = ProxyArray(wp.zeros(num_envs, dtype=wp.vec4f, device="cpu"))
    term._velocity = term._state.torch[:, 2:]
    term._step_launch = term._raw = None

    manager_actions = torch.zeros(num_envs, 5)
    manager_actions[:, 2:4] = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [3.0, 0.0]])
    term.process_actions(manager_actions[:, 2:4])
    expected = clamp_gaze(manager_actions[:, 2:4].clamp(-1, 1) * 0.1, 1.0)
    torch.testing.assert_close(term.gaze, expected)
    torch.testing.assert_close(term.velocity, expected / 0.5)
    torch.testing.assert_close(angles.torch, expected * 20.0)
    assert set_angles[-1] is angles.torch
    assert term.raw_actions.data_ptr() == manager_actions[:, 2:4].data_ptr()

    supplied = torch.tensor([[-1.0, 0.0]]).repeat(num_envs, 1)
    term.process_actions(supplied)
    torch.testing.assert_close(term.gaze, clamp_gaze(expected - 0.1 * supplied.abs(), 1.0))
    assert term.raw_actions is supplied
    torch.testing.assert_close(
        gaze_state(SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))), term.state
    )

    # A gaze set on the camera directly is where the next step starts.
    angles.torch.copy_(torch.tensor([[10.0, 0.0]]).repeat(num_envs, 1))
    term.process_actions(torch.zeros(num_envs, 2))
    torch.testing.assert_close(term.gaze, torch.tensor([[0.5, 0.0]]).repeat(num_envs, 1))


@pytest.mark.parametrize(("ids", "random_start"), [(None, False), ([3, 0, 7], True)])
def test_gaze_reset_kernel_matches_the_torch_reset(ids, random_start):
    count, radius = 10, 23.6
    generator = torch.Generator().manual_seed(6)
    angles, state = (torch.randn(count, size, generator=generator) for size in (2, 4))
    selected = torch.arange(count) if ids is None else torch.tensor(ids)
    start = torch.randn(len(selected), 2, generator=generator) if random_start else torch.zeros(len(selected), 2)
    expected_angles, expected_state = angles.clone(), state.clone()
    # The torch reset the kernel replaces: the gaze at its start, at rest.
    expected_state[selected] = torch.cat((start, torch.zeros_like(start)), dim=-1)
    expected_angles[selected] = start * radius

    with torch_stream("cpu"):
        wp.launch(
            gaze_reset_kernel,
            dim=len(selected),
            inputs=[
                wp.from_torch(selected.to(torch.int32)) if ids is not None else wp.zeros(1, dtype=wp.int32),
                int(ids is not None),
                wp.from_torch(start, dtype=wp.vec2f) if random_start else wp.zeros(1, dtype=wp.vec2f),
                int(random_start),
                radius,
            ],
            outputs=[wp.from_torch(angles, dtype=wp.vec2f), wp.from_torch(state, dtype=wp.vec4f)],
        )
    torch.testing.assert_close(angles, expected_angles, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_fixed_mount_teacher_rejects_only_turns_after_capture():
    """The fixed-mount teacher aims from the captured mount while the camera has turned its mount as many times as
    when the term captured it, and refuses once a later turn makes the capture stale."""
    term = object.__new__(GazeAction)
    term.tracked_mount = False
    term._mount_turns = 2
    term._fixed_mount = "captured"
    term._asset = SimpleNamespace(mount_turns=2)
    assert GazeAction.teacher_mounts(term) == "captured"
    term._asset.mount_turns = 3
    with pytest.raises(RuntimeError, match="mount='tracked'"):
        GazeAction.teacher_mounts(term)
