# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused PhysX dual actuator-dispatch tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import warp as wp
from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl


class _RecordingView:
    def __init__(self) -> None:
        self.effort = None
        self.position = None
        self.velocity = None

    def set_dof_actuation_forces(self, values, indices) -> None:
        self.effort = values

    def set_dof_position_targets(self, values, indices) -> None:
        self.position = values

    def set_dof_velocity_targets(self, values, indices) -> None:
        self.velocity = values


def _buffer(value: float) -> wp.array:
    return wp.full((1, 2), value, dtype=wp.float32, device="cpu")


def test_submit_commands_uses_processed_lab_buffers_for_standard_path() -> None:
    """The standard path must submit processed effort and implicit drive targets."""
    view = _RecordingView()
    articulation = SimpleNamespace(
        _has_newton_actuators=False,
        _has_implicit_actuators=True,
        data=SimpleNamespace(has_joint_ordering=False),
        root_view=view,
        _ALL_INDICES=wp.array([0], dtype=wp.int32, device="cpu"),
    )
    collection = SimpleNamespace(
        _joint_effort_target_sim=_buffer(1.0),
        _joint_pos_target_sim=_buffer(2.0),
        _joint_vel_target_sim=_buffer(3.0),
    )
    control = object.__new__(PhysxActuatorControl)
    control._articulation = articulation

    control.submit_commands(collection)

    assert view.effort is collection._joint_effort_target_sim
    assert view.position is collection._joint_pos_target_sim
    assert view.velocity is collection._joint_vel_target_sim


def test_submit_commands_uses_native_effort_and_public_targets_for_newton_path() -> None:
    """The Newton path must submit wrapper effort while retaining public position and velocity targets."""
    view = _RecordingView()
    native_effort = _buffer(4.0)
    articulation = SimpleNamespace(
        _has_newton_actuators=True,
        _has_implicit_actuators=True,
        _physx_actuator_wrapper=SimpleNamespace(joint_f_2d=native_effort),
        data=SimpleNamespace(has_joint_ordering=False),
        root_view=view,
        _ALL_INDICES=wp.array([0], dtype=wp.int32, device="cpu"),
    )
    collection = SimpleNamespace(
        _joint_pos_target=_buffer(5.0),
        _joint_vel_target=_buffer(6.0),
    )
    control = object.__new__(PhysxActuatorControl)
    control._articulation = articulation

    control.submit_commands(collection)

    assert view.effort is native_effort
    assert view.position is collection._joint_pos_target
    assert view.velocity is collection._joint_vel_target


def test_compute_native_actuators_refreshes_nonidentity_public_joint_state() -> None:
    """The native controller must observe the latest reordered PhysX joint state."""
    refresh_position = Mock()
    refresh_velocity = Mock()
    runtime = SimpleNamespace(compute=Mock())
    articulation = SimpleNamespace(
        data=SimpleNamespace(has_joint_ordering=True),
        _data=SimpleNamespace(_refresh_joint_pos=refresh_position, _refresh_joint_vel=refresh_velocity),
    )
    collection = SimpleNamespace()
    control = object.__new__(PhysxActuatorControl)
    control._articulation = articulation
    control._native_actuator_path_active = True
    control._actuator_runtime = runtime

    assert control.compute_native_actuators(collection, 0.01)

    refresh_position.assert_called_once_with()
    refresh_velocity.assert_called_once_with()
    runtime.compute.assert_called_once_with(collection, 0.01)
