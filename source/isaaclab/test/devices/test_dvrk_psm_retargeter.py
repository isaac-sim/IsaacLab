# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore private usage of variables warning.
# pyright: reportPrivateUsage=none

from __future__ import annotations

import builtins

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.isaacsim_ci

from isaaclab.devices import DeviceBase, RetargeterBase
from isaaclab.devices.openxr.retargeters import (
    DVRKPSMRetargeter,
    DVRKPSMRetargeterCfg,
    DVRKPSMSideRetargeterCfg,
)


def _side_cfg(
    *,
    x_offset: float,
    home_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    orientation_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    initial_closedness: float = 0.0,
) -> DVRKPSMSideRetargeterCfg:
    return DVRKPSMSideRetargeterCfg(
        home_position=(x_offset, 0.0, 0.2),
        home_orientation=home_orientation,
        workspace_lower=(x_offset - 0.1, -0.1, 0.1),
        workspace_upper=(x_offset + 0.1, 0.1, 0.3),
        translation_scale=1.0,
        orientation_offset=orientation_offset,
        jaw_open=(-0.5, 0.5),
        jaw_closed=(-0.09, 0.09),
        initial_closedness=initial_closedness,
        clutch_threshold=0.5,
        trigger_deadband=0.05,
        opening_intent_duration_s=0.1,
    )


def _cfg(*, sim_device: str = "cpu") -> DVRKPSMRetargeterCfg:
    return DVRKPSMRetargeterCfg(
        sim_device=sim_device,
        left=_side_cfg(x_offset=-0.2),
        right=_side_cfg(x_offset=0.2),
    )


def _controller(
    position=(0.0, 0.0, 0.0),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    *,
    trigger=0.0,
    squeeze=1.0,
    pose_valid=1.0,
) -> np.ndarray:
    pose = np.asarray((*position, *orientation_wxyz), dtype=np.float64)
    inputs = np.asarray((0.0, 0.0, trigger, squeeze, 0.0, 0.0, pose_valid), dtype=np.float64)
    return np.stack((pose, inputs))


def _both(left: np.ndarray, right: np.ndarray) -> dict:
    return {
        DeviceBase.TrackingTarget.CONTROLLER_LEFT: left,
        DeviceBase.TrackingTarget.CONTROLLER_RIGHT: right,
    }


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Build a rotation matrix without using Isaac Teleop quaternion helpers."""
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a test-oracle matrix to wxyz using an eigenvector construction."""
    xx, xy, xz = rotation[0]
    yx, yy, yz = rotation[1]
    zx, zy, zz = rotation[2]
    symmetric = (
        np.array(
            (
                (xx - yy - zz, xy + yx, xz + zx, zy - yz),
                (xy + yx, yy - xx - zz, yz + zy, xz - zx),
                (xz + zx, yz + zy, zz - xx - yy, yx - xy),
                (zy - yz, xz - zx, yx - xy, xx + yy + zz),
            )
        )
        / 3.0
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    xyzw = eigenvectors[:, np.argmax(eigenvalues)]
    if xyzw[3] < 0.0:
        xyzw *= -1.0
    return float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])


def _quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> np.ndarray:
    """Convert adapter output to a matrix independently of its implementation."""
    x, y, z, w = quaternion.cpu().numpy().astype(np.float64)
    norm = np.linalg.norm((x, y, z, w))
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


@pytest.fixture
def retargeter() -> DVRKPSMRetargeter:
    pytest.importorskip("isaacteleop.retargeters.DVRK.control")
    return DVRKPSMRetargeter(_cfg())


def test_lazy_dependency_error_is_targeted(monkeypatch):
    real_import = builtins.__import__

    def import_with_missing_control(name, *args, **kwargs):
        if name == "isaacteleop.retargeters.DVRK.control":
            raise ImportError("missing test dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_control)
    with pytest.raises(ModuleNotFoundError, match="Isaac Teleop.*PR 769"):
        DVRKPSMRetargeter(_cfg())


def test_retargeter_requests_only_motion_controller_data(retargeter):
    assert retargeter.get_requirements() == [RetargeterBase.Requirement.MOTION_CONTROLLER]


def test_inactive_session_holds_then_emits_stable_18d_abi(retargeter):
    home = retargeter.action.clone()
    pre_start = _both(
        _controller(position=(4.0, 5.0, 6.0), trigger=1.0),
        _controller(position=(-4.0, -5.0, -6.0), trigger=1.0),
    )
    assert torch.equal(retargeter.retarget(pre_start), home)

    retargeter.start()
    origins = _both(_controller(position=(0.3, 0.1, 0.4)), _controller(position=(-0.3, 0.1, 0.4)))
    assert torch.equal(retargeter.retarget(origins), home)

    moved = _both(
        _controller(position=(0.32, 0.09, 0.43), trigger=1.0),
        _controller(position=(-0.3, 0.1, 0.4)),
    )
    action = retargeter.retarget(moved)

    assert action.shape == (18,)
    assert action.dtype == torch.float32
    assert action.device == torch.device("cpu")
    assert action.is_contiguous()
    assert torch.isfinite(action).all()
    # [left xyz, left xyzw, left jaws, right xyz, right xyzw, right jaws]
    torch.testing.assert_close(action[:3], torch.tensor((-0.18, -0.01, 0.23)))
    torch.testing.assert_close(action[3:7], torch.tensor((0.0, 0.0, 0.0, 1.0)))
    torch.testing.assert_close(action[7:9], torch.tensor((-0.09, 0.09)))
    torch.testing.assert_close(action[9:12], torch.tensor((0.2, 0.0, 0.2)))
    torch.testing.assert_close(action[12:16], torch.tensor((0.0, 0.0, 0.0, 1.0)))
    torch.testing.assert_close(action[16:18], torch.tensor((-0.5, 0.5)))


def test_non_identity_wxyz_inputs_follow_independent_non_commuting_rotation_oracle():
    pytest.importorskip("isaacteleop.retargeters.DVRK.control")
    home_rotation = _rotation((1.0, 1.0, 0.0), np.deg2rad(23.0))
    offset_rotation = _rotation((0.0, 1.0, 0.0), np.deg2rad(67.0))
    origin_rotation = _rotation((0.0, 0.0, 1.0), np.deg2rad(-31.0))
    relative_rotation = _rotation((1.0, 0.0, 0.0), np.deg2rad(41.0))
    current_rotation = origin_rotation @ relative_rotation
    assert not np.allclose(offset_rotation @ relative_rotation, relative_rotation @ offset_rotation)

    cfg = DVRKPSMRetargeterCfg(
        sim_device="cpu",
        left=_side_cfg(
            x_offset=-0.2,
            home_orientation=_matrix_to_quaternion_wxyz(home_rotation)[1:]
            + _matrix_to_quaternion_wxyz(home_rotation)[:1],
            orientation_offset=_matrix_to_quaternion_wxyz(offset_rotation)[1:]
            + _matrix_to_quaternion_wxyz(offset_rotation)[:1],
        ),
        right=_side_cfg(x_offset=0.2),
    )
    retargeter = DVRKPSMRetargeter(cfg)
    retargeter.start()
    retargeter.retarget(
        _both(
            _controller(orientation_wxyz=_matrix_to_quaternion_wxyz(origin_rotation)),
            _controller(),
        )
    )
    action = retargeter.retarget(
        _both(
            _controller(orientation_wxyz=_matrix_to_quaternion_wxyz(current_rotation)),
            _controller(),
        )
    )

    expected_rotation = home_rotation @ offset_rotation @ relative_rotation @ offset_rotation.T
    np.testing.assert_allclose(_quaternion_xyzw_to_matrix(action[3:7]), expected_rotation, atol=2.0e-6)
    np.testing.assert_allclose(_quaternion_xyzw_to_matrix(action[12:16]), np.eye(3), atol=2.0e-6)


def test_differently_placed_sides_clip_to_independent_world_bounds(retargeter):
    retargeter.start()
    origins = _both(_controller(), _controller())
    retargeter.retarget(origins)

    action = retargeter.retarget(
        _both(_controller(position=(10.0, 10.0, 10.0)), _controller(position=(-10.0, -10.0, -10.0)))
    )
    torch.testing.assert_close(action[:3], torch.tensor((-0.1, 0.1, 0.3)))
    torch.testing.assert_close(action[9:12], torch.tensor((0.1, -0.1, 0.1)))


@pytest.mark.parametrize(
    "bad_left",
    (
        None,
        np.zeros((1, 7)),
        _controller(pose_valid=0.0),
        _controller(orientation_wxyz=(0.0, 0.0, 0.0, 0.0)),
        _controller(position=(np.nan, 0.0, 0.0)),
        _controller(trigger=np.inf),
        [[10**1000] * 7, [0.0] * 7],
    ),
)
def test_invalid_left_holds_full_side_while_right_remains_independent(retargeter, bad_left):
    retargeter.start()
    origins = _both(_controller(), _controller())
    retargeter.retarget(origins)
    previous = retargeter.retarget(
        _both(_controller(position=(0.02, 0.0, 0.0), trigger=1.0), _controller(position=(0.01, 0.0, 0.0)))
    )

    data = {DeviceBase.TrackingTarget.CONTROLLER_RIGHT: _controller(position=(0.03, 0.0, 0.0))}
    if bad_left is not None:
        data[DeviceBase.TrackingTarget.CONTROLLER_LEFT] = bad_left
    action = retargeter.retarget(data)

    assert torch.equal(action[:9], previous[:9])
    assert not torch.equal(action[9:12], previous[9:12])
    assert torch.isfinite(action).all()


@pytest.mark.parametrize("loss", ("squeeze", "tracking"))
def test_squeeze_release_and_tracking_loss_hold_then_capture_fresh_origin(retargeter, loss):
    retargeter.start()
    retargeter.retarget(_both(_controller(), _controller()))
    moved = retargeter.retarget(_both(_controller(position=(0.02, 0.0, 0.0), trigger=1.0), _controller()))
    torch.testing.assert_close(moved[:3], torch.tensor((-0.18, 0.0, 0.2)))
    torch.testing.assert_close(moved[7:9], torch.tensor((-0.09, 0.09)))

    loss_sample = (
        _controller(position=(0.08, 0.0, 0.0), trigger=0.0, squeeze=0.0)
        if loss == "squeeze"
        else _controller(position=(0.08, 0.0, 0.0), trigger=0.0, pose_valid=0.0)
    )
    held = retargeter.retarget(_both(loss_sample, _controller()))
    assert torch.equal(held[:9], moved[:9])

    recaptured = retargeter.retarget(_both(_controller(position=(0.08, 0.0, 0.0), trigger=0.0), _controller()))
    assert torch.equal(recaptured[:9], moved[:9])
    after_fresh_origin = retargeter.retarget(_both(_controller(position=(0.09, 0.0, 0.0), trigger=0.0), _controller()))
    torch.testing.assert_close(after_fresh_origin[:3], torch.tensor((-0.17, 0.0, 0.2)))
    assert torch.equal(after_fresh_origin[7:9], moved[7:9])


def test_deliberate_opening_waits_for_dwell_then_emits_configured_targets(mocker):
    pytest.importorskip("isaacteleop.retargeters.DVRK.control")
    cfg = DVRKPSMRetargeterCfg(
        sim_device="cpu",
        left=_side_cfg(x_offset=-0.2, initial_closedness=1.0),
        right=_side_cfg(x_offset=0.2, initial_closedness=1.0),
    )
    retargeter = DVRKPSMRetargeter(cfg)
    mocker.patch(
        "isaaclab.devices.openxr.retargeters.manipulator.dvrk_psm_retargeter.time.monotonic_ns",
        side_effect=(0, 50_000_000, 100_000_000, 160_000_000),
    )
    retargeter.start()

    closed = retargeter.retarget(_both(_controller(trigger=1.0), _controller(trigger=1.0)))
    first_opening_sample = retargeter.retarget(_both(_controller(trigger=0.0), _controller(trigger=1.0)))
    still_dwelling = retargeter.retarget(_both(_controller(trigger=0.0), _controller(trigger=1.0)))
    opened = retargeter.retarget(_both(_controller(trigger=0.0), _controller(trigger=1.0)))

    expected_closed = torch.tensor((-0.09, 0.09))
    expected_open = torch.tensor((-0.5, 0.5))
    torch.testing.assert_close(closed[7:9], expected_closed)
    torch.testing.assert_close(first_opening_sample[7:9], expected_closed)
    torch.testing.assert_close(still_dwelling[7:9], expected_closed)
    torch.testing.assert_close(opened[7:9], expected_open)
    torch.testing.assert_close(opened[16:18], expected_closed)


def test_configured_non_cpu_tensor_device_is_honoured():
    pytest.importorskip("isaacteleop.retargeters.DVRK.control")
    retargeter = DVRKPSMRetargeter(_cfg(sim_device="meta"))

    action = retargeter.action

    assert action.shape == (18,)
    assert action.dtype == torch.float32
    assert action.device == torch.device("meta")
    assert action.is_contiguous()


def test_stop_disengages_reset_preserves_activity_and_requires_fresh_origins(retargeter):
    retargeter.start()
    retargeter.retarget(_both(_controller(), _controller()))
    moved = retargeter.retarget(
        _both(_controller(position=(0.03, 0.0, 0.0), trigger=1.0), _controller(position=(-0.03, 0.0, 0.0)))
    )

    retargeter.stop()
    assert not retargeter.session_active
    assert torch.equal(retargeter.action, moved)
    assert torch.equal(
        retargeter.retarget(_both(_controller(position=(8.0, 8.0, 8.0)), _controller(position=(8.0, 8.0, 8.0)))),
        moved,
    )

    retargeter.start()
    assert torch.equal(retargeter.retarget(_both(_controller(), _controller())), moved)
    retargeter.reset()
    assert retargeter.session_active
    reset_action = retargeter.action
    torch.testing.assert_close(reset_action[:3], torch.tensor((-0.2, 0.0, 0.2)))
    torch.testing.assert_close(reset_action[7:9], torch.tensor((-0.5, 0.5)))
    torch.testing.assert_close(reset_action[9:12], torch.tensor((0.2, 0.0, 0.2)))
    assert torch.equal(
        retargeter.retarget(_both(_controller(position=(2.0, 2.0, 2.0)), _controller(position=(-2.0, -2.0, -2.0)))),
        reset_action,
    )


def test_jaw_clock_uses_independent_high_water_marks_and_clears_on_lifecycle(retargeter, mocker):
    left_step = mocker.spy(retargeter._left.jaw, "step")
    right_step = mocker.spy(retargeter._right.jaw, "step")
    clock = mocker.patch(
        "isaaclab.devices.openxr.retargeters.manipulator.dvrk_psm_retargeter.time.monotonic_ns",
        side_effect=(100, 100, 90, 150, 200, 250),
    )
    sample = _both(_controller(), _controller())

    for _ in range(4):
        retargeter.retarget(sample)
    assert [call.kwargs["dt_seconds"] for call in left_step.call_args_list] == [0.0, 0.0, 0.0, 50e-9]
    assert [call.kwargs["dt_seconds"] for call in right_step.call_args_list] == [0.0, 0.0, 0.0, 50e-9]

    retargeter.reset()
    retargeter.retarget(sample)
    assert left_step.call_args.kwargs["dt_seconds"] == 0.0
    assert right_step.call_args.kwargs["dt_seconds"] == 0.0
    retargeter.stop()
    retargeter.retarget(sample)
    assert left_step.call_args.kwargs["dt_seconds"] == 0.0
    assert right_step.call_args.kwargs["dt_seconds"] == 0.0
    assert clock.call_count == 6
