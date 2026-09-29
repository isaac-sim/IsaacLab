# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for PhysX Newton-actuator path preparation."""

from types import SimpleNamespace

import warp as wp
from isaaclab_physx.assets.articulation import actuator_control
from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.actuators.newton import NewtonActuatorAdapter, PhysxActuatorWrapper


def test_prepare_native_actuators_does_not_zero_solver_gains(monkeypatch):
    """Leave solver gains untouched until collection construction resolves actuator defaults."""
    joint_buffer = SimpleNamespace(warp=wp.zeros((1, 1), dtype=wp.float32, device="cpu"))
    collection = SimpleNamespace(
        target_command=SimpleNamespace(position=joint_buffer, velocity=joint_buffer, effort=joint_buffer)
    )
    gain_writes = []
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
        joint_names=["joint"],
        num_instances=1,
        num_joints=1,
        device="cpu",
        _data=SimpleNamespace(joint_pos=joint_buffer, joint_vel=joint_buffer),
        write_joint_stiffness_to_sim_index=lambda **_: gain_writes.append("stiffness"),
        write_joint_damping_to_sim_index=lambda **_: gain_writes.append("damping"),
    )
    wrapper = SimpleNamespace()
    adapter = SimpleNamespace(joint_indices=wp.array([0], dtype=wp.int32), finalize=lambda _: None)
    monkeypatch.setattr(actuator_control, "find_first_matching_prim", lambda _: None)
    monkeypatch.setattr(PhysxActuatorWrapper, "create", lambda **_: wrapper)
    monkeypatch.setattr(NewtonActuatorAdapter, "from_usd", lambda **_: adapter)

    native_groups = PhysxActuatorControl(articulation).prepare_native_actuators(
        collection,
        {"explicit": IdealPDActuatorCfg(joint_names_expr=["joint"], stiffness=None, damping=None)},
    )

    assert native_groups == {"explicit"}
    assert gain_writes == []


def test_prepare_native_actuators_leaves_implicit_only_articulation_on_standard_path(monkeypatch):
    """Keep implicit-only articulations on the unchanged solver-drive path."""
    runtime_prepare_calls = []
    runtime = SimpleNamespace(
        prepare=lambda *args, **kwargs: runtime_prepare_calls.append(True), wrapper=None, adapter=None
    )
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
    )
    monkeypatch.setattr(actuator_control, "PhysxActuatorRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(actuator_control, "find_first_matching_prim", lambda _: None)

    control = PhysxActuatorControl(articulation)
    native_groups = control.prepare_native_actuators(
        collection=None,
        actuator_cfgs={"implicit": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
    )

    assert native_groups == set()
    assert not control.native_actuator_path_active
    assert not articulation._has_newton_actuators
    assert runtime_prepare_calls == []
