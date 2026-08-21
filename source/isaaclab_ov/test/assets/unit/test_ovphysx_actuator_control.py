# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for OVPhysX actuator command and native-runtime adaptation."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import warp as wp
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.assets.articulation import actuator_control as actuator_control_module
from isaaclab_ov.assets.articulation.actuator_control import OvPhysxActuatorControl

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg

pytestmark = pytest.mark.unit


class _RecordingView:
    """Capture values written to each OVPhysX tensor type."""

    def __init__(self) -> None:
        self.values = {}
        self.kwargs = {}

    def set_attribute(self, tensor_type, values, **kwargs) -> None:
        self.values[tensor_type] = values
        self.kwargs[tensor_type] = kwargs


def _buffer(value: float) -> wp.array:
    return wp.full((1, 2), value, dtype=wp.float32, device="cpu")


def test_prepare_and_finalize_native_actuators_own_runtime_boundary(monkeypatch) -> None:
    """OV preparation must classify groups and expose the shared runtime selection."""
    calls = []
    wrapper = SimpleNamespace()
    native_actuator = object()
    adapter = SimpleNamespace(actuators=[native_actuator])
    stage = object()

    class _Runtime:
        def __init__(self, owner, *, logger):
            calls.append(("construct", owner))
            self.wrapper = wrapper
            self.adapter = adapter

        def prepare(self, collection, **kwargs) -> None:
            calls.append(("prepare", collection, kwargs))

        def finalize(self, collection) -> None:
            calls.append(("finalize", collection))

    monkeypatch.setattr(actuator_control_module, "_validate_newton_native_actuator_cfgs", lambda cfgs: None)
    monkeypatch.setattr(actuator_control_module, "find_first_matching_prim", lambda path: None)
    monkeypatch.setattr(actuator_control_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(actuator_control_module, "PhysxActuatorRuntime", _Runtime)
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
        num_instances=2,
        num_joints=3,
        num_fixed_tendons=0,
        device="cpu",
    )
    collection = SimpleNamespace()
    control = OvPhysxActuatorControl(articulation)

    native_groups = control.prepare_native_actuators(
        collection,
        {
            "implicit": ImplicitActuatorCfg(joint_names_expr=["joint_0"], stiffness=2.0, damping=0.2),
            "explicit": IdealPDActuatorCfg(joint_names_expr=["joint_[12]"], stiffness=3.0, damping=0.3),
        },
    )
    selection = control.finalize_native_actuators(collection)

    assert native_groups == {"explicit"}
    assert control.native_actuator_path_active
    assert articulation._has_newton_actuators
    assert articulation._physx_actuator_wrapper is wrapper
    assert articulation.newton_actuator_adapter is adapter
    assert selection is not None
    assert selection.actuators == [native_actuator]
    assert selection.view.world_count == 2
    assert calls == [
        ("construct", articulation),
        (
            "prepare",
            collection,
            {"stage": stage, "articulation_prim_path": None, "adapt_usd_actuators": True},
        ),
        ("finalize", collection),
    ]


def test_submit_commands_uses_processed_lab_buffers_without_native_runtime() -> None:
    """The standard path must submit applied effort and implicit drive targets."""
    view = _RecordingView()
    articulation = SimpleNamespace(
        _can_write_effort=True,
        _can_write_pos_target=True,
        _can_write_vel_target=True,
        _has_implicit_actuators=True,
        data=SimpleNamespace(has_joint_ordering=False),
        _root_view=view,
    )
    collection = SimpleNamespace(
        _applied_effort=_buffer(1.0),
        _joint_pos_target=_buffer(2.0),
        _joint_vel_target=_buffer(3.0),
    )
    control = object.__new__(OvPhysxActuatorControl)
    control._articulation = articulation
    control._actuator_runtime = None

    control.submit_commands(collection)

    assert view.values[TT.DOF_ACTUATION_FORCE] is collection._applied_effort
    assert view.values[TT.DOF_POSITION_TARGET] is collection._joint_pos_target
    assert view.values[TT.DOF_VELOCITY_TARGET] is collection._joint_vel_target


def test_submit_commands_uses_native_raw_effort_without_double_counting_implicit_pd() -> None:
    """A native runtime must replace telemetry effort while retaining public drive targets."""
    view = _RecordingView()
    native_effort = _buffer(4.0)
    articulation = SimpleNamespace(
        _can_write_effort=True,
        _can_write_pos_target=True,
        _can_write_vel_target=True,
        _has_implicit_actuators=True,
        data=SimpleNamespace(has_joint_ordering=False),
        _root_view=view,
    )
    collection = SimpleNamespace(
        _applied_effort=_buffer(1.0),
        _joint_pos_target=_buffer(5.0),
        _joint_vel_target=_buffer(6.0),
    )
    control = object.__new__(OvPhysxActuatorControl)
    control._articulation = articulation
    control._actuator_runtime = SimpleNamespace(wrapper=SimpleNamespace(joint_f_2d=native_effort))

    control.submit_commands(collection)

    assert view.values[TT.DOF_ACTUATION_FORCE] is native_effort
    assert view.values[TT.DOF_POSITION_TARGET] is collection._joint_pos_target
    assert view.values[TT.DOF_VELOCITY_TARGET] is collection._joint_vel_target


def test_native_compute_refreshes_owned_state_before_controller() -> None:
    """The native controller must observe fresh OVPhysX position and velocity shadows."""
    ordered_calls = Mock()
    articulation = SimpleNamespace(
        _data=SimpleNamespace(
            _refresh_joint_pos=ordered_calls.refresh_position,
            _refresh_joint_vel=ordered_calls.refresh_velocity,
        )
    )
    control = object.__new__(OvPhysxActuatorControl)
    control._articulation = articulation
    control._native_actuator_path_active = True
    control._actuator_runtime = SimpleNamespace(compute=ordered_calls.compute)
    collection = SimpleNamespace()

    assert control.compute_native_actuators(collection, 0.01)

    assert ordered_calls.mock_calls == [
        call.refresh_position(),
        call.refresh_velocity(),
        call.compute(collection, 0.01),
    ]


def test_native_reset_delegates_exact_environment_selector() -> None:
    """Selective reset must pass the caller's environment selector unchanged."""
    runtime = Mock()
    env_ids = wp.array([1, 0], dtype=wp.int32, device="cpu")
    control = object.__new__(OvPhysxActuatorControl)
    control._native_actuator_path_active = True
    control._actuator_runtime = runtime

    control.reset_native_actuators(env_ids)

    runtime.reset.assert_called_once_with(env_ids)


def test_stage_user_command_converts_partial_environment_selector_to_sim_indices() -> None:
    """Partial public commands must use the OVPhysX int32 simulator selector."""
    view = _RecordingView()
    user = _buffer(7.0)
    backend = _buffer(0.0)
    sim_ids = wp.array([1], dtype=wp.int32, device="cpu")
    articulation = SimpleNamespace(
        _can_write_pos_target=True,
        _joint_pos_target_backend=backend,
        _get_backend_ordered_joint_buffer=lambda values, staging: values,
        _get_sim_env_ids=lambda values: sim_ids,
        _root_view=view,
    )
    collection = SimpleNamespace(_joint_pos_target=user)
    control = object.__new__(OvPhysxActuatorControl)
    control._articulation = articulation

    control.stage_user_command(
        "position",
        collection,
        env_ids=wp.array([1], dtype=wp.int64, device="cpu"),
        joint_ids=None,
        env_mask=None,
        joint_mask=None,
    )

    assert view.values[TT.DOF_POSITION_TARGET] is user
    assert view.kwargs[TT.DOF_POSITION_TARGET]["indices"] is sim_ids
