# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused Newton actuator-adaptation and telemetry tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
from isaaclab_newton.assets.articulation.articulation import _configure_builder_joint_target_modes
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import JointTargetMode, JointType, Model, ModelBuilder

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.actuators.newton.adapter import resolve_actuator_component
from isaaclab.actuators.newton.kernels import sync_torque_telemetry
from isaaclab.assets import ArticulationCfg


def _target_mode_builder() -> ModelBuilder:
    builder = ModelBuilder()
    inertia = wp.mat33(1.0)
    left_link = builder.add_link(mass=1.0, inertia=inertia, label="/World/Robot/left_link")
    left_joint = builder.add_joint_revolute(-1, left_link, label="/World/Robot/left_joint")
    right_link = builder.add_link(mass=1.0, inertia=inertia, label="/World/Robot/right_link")
    right_joint = builder.add_joint_revolute(left_link, right_link, label="/World/Robot/right_joint")
    builder.add_articulation([left_joint, right_joint], label="/World/Robot")
    builder.articulation_label = ["/World/Robot"]
    builder.joint_target_mode = [int(JointTargetMode.NONE), int(JointTargetMode.NONE)]
    builder.joint_target_ke = [0.0, 0.0]
    builder.joint_target_kd = [0.0, 0.0]
    return builder


@pytest.mark.parametrize(
    ("actuator_cfg", "imported_ke", "imported_kd", "initial_modes", "joint_types", "expected_modes"),
    [
        pytest.param(
            ImplicitActuatorCfg(joint_names_expr=[".*_joint"], stiffness=None, damping=None),
            [8.0, 0.0],
            [0.0, 3.0],
            [0, 0],
            None,
            [1, 2],
            id="imported-none-gains",
        ),
        pytest.param(
            ImplicitActuatorCfg(joint_names_expr=[".*_joint"], stiffness=0.0, damping=0.0),
            [8.0, 8.0],
            [3.0, 3.0],
            [2, 2],
            None,
            [4, 4],
            id="zero-gains",
        ),
        pytest.param(
            ImplicitActuatorCfg(joint_names_expr=[".*_joint"], stiffness=8.0, damping=3.0),
            [0.0, 0.0],
            [0.0, 0.0],
            [0, 0],
            None,
            [3, 3],
            id="both-gains",
        ),
        pytest.param(
            IdealPDActuatorCfg(joint_names_expr=[".*_joint"], stiffness=8.0, damping=3.0),
            [0.0, 0.0],
            [0.0, 0.0],
            [0, 0],
            None,
            [4, 4],
            id="explicit-effort",
        ),
        pytest.param(
            ImplicitActuatorCfg(joint_names_expr=[".*_joint"], stiffness=8.0, damping=3.0),
            [0.0, 0.0],
            [0.0, 0.0],
            [1, 2],
            [JointType.FREE, JointType.FIXED],
            [1, 2],
            id="free-fixed-excluded",
        ),
        pytest.param(
            ImplicitActuatorCfg(joint_names_expr=["left_joint"], stiffness=8.0, damping=0.0),
            [0.0, 0.0],
            [0.0, 0.0],
            [0, 2],
            None,
            [1, 2],
            id="unconfigured-dof-unchanged",
        ),
        pytest.param(
            ImplicitActuatorCfg(
                joint_names_expr=[".*_joint"],
                stiffness={"left_joint": 10.0},
                damping={"right_joint": 2.0},
            ),
            [0.0, 0.0],
            [0.0, 0.0],
            [0, 0],
            None,
            [1, 2],
            id="sparse-position-velocity",
        ),
    ],
)
def test_builder_target_modes_cover_all_adaptation_branches(
    monkeypatch,
    actuator_cfg,
    imported_ke: list[float],
    imported_kd: list[float],
    initial_modes: list[int],
    joint_types,
    expected_modes: list[int],
) -> None:
    """Every builder branch must assign literal modes without changing excluded or unmatched DOFs."""
    cfg = ArticulationCfg(prim_path="/World/Robot", actuators={"joints": actuator_cfg})
    monkeypatch.setattr(
        "isaaclab_newton.assets.articulation.articulation._resolve_articulation_root_prim_path_expr",
        lambda _cfg: "/World/Robot",
    )
    builder = _target_mode_builder()
    builder.joint_target_ke = imported_ke
    builder.joint_target_kd = imported_kd
    builder.joint_target_mode = initial_modes
    if joint_types is not None:
        builder.joint_type = joint_types

    _configure_builder_joint_target_modes(builder, cfg)

    assert builder.joint_target_mode == expected_modes


def test_prepare_native_actuators_activates_only_explicit_groups_without_gain_writes(monkeypatch) -> None:
    """Newton adaptation must select explicit groups without clobbering imported solver gains."""
    articulation = SimpleNamespace(_sim_cfg=SimpleNamespace(use_newton_actuators=True))
    activations = []
    monkeypatch.setattr(
        SimulationManager, "activate_newton_actuator_path", classmethod(lambda cls: activations.append(1))
    )

    groups = NewtonActuatorControl(articulation).prepare_native_actuators(
        collection=None,
        actuator_cfgs={
            "implicit": ImplicitActuatorCfg(joint_names_expr=["left_joint"], stiffness=10.0, damping=1.0),
            "explicit": IdealPDActuatorCfg(joint_names_expr=["right_joint"], stiffness=10.0, damping=1.0),
        },
    )

    assert groups == {"explicit"}
    assert activations == [1]
    assert articulation._has_newton_actuators


def test_resolve_actuator_component_rejects_ambiguous_clamping_owner() -> None:
    """Actuator adaptation must reject two clamping components exposing the same parameter."""
    actuator = SimpleNamespace(
        controller=SimpleNamespace(kp=1.0),
        delay=None,
        clamping=[SimpleNamespace(limit=1.0), SimpleNamespace(limit=2.0)],
    )

    with pytest.raises(ValueError, match="Ambiguous clamping parameter 'limit'"):
        resolve_actuator_component(actuator, "clamping", "limit")


@pytest.mark.parametrize(
    ("layout", "expected_offset"),
    [
        (SimpleNamespace(offset=10, slice=slice(3, 5), indices=None), 13),
        (SimpleNamespace(offset=10, slice=None, indices=wp.array([4, 7], dtype=wp.int32, device="cpu")), 14),
    ],
)
def test_articulation_dof_offset_accounts_for_each_view_selection_layout(layout, expected_offset: int) -> None:
    """Heterogeneous articulation bindings must offset native actuators by their selected model DOFs."""
    control = object.__new__(NewtonActuatorControl)
    control._articulation = SimpleNamespace(
        _root_view=SimpleNamespace(frequency_layouts={Model.AttributeFrequency.JOINT_DOF: layout})
    )

    assert control._joint_dof_offset() == expected_offset


def test_native_actuator_reset_delegates_selected_environments_to_adapter(monkeypatch) -> None:
    """A partial articulation reset must preserve state in unselected native-actuator environments."""
    reset_calls = []
    control = object.__new__(NewtonActuatorControl)
    control._native_actuator_path_active = True
    monkeypatch.setattr(
        SimulationManager, "_adapter", SimpleNamespace(reset=lambda env_ids: reset_calls.append(env_ids))
    )
    env_ids = [1]

    control.reset_native_actuators(env_ids)

    assert reset_calls == [env_ids]


@pytest.mark.parametrize("has_ordering", [False, True])
def test_torque_telemetry_preserves_public_joint_order(has_ordering: bool) -> None:
    """Newton telemetry must map backend buffers exactly once when public ordering is active."""
    zeros = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    effort_limit = wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu")
    implicit = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    effort = wp.array([[100.0, 200.0, 300.0]], dtype=wp.float32, device="cpu")
    computed_source = wp.array([[10.0, 20.0, 30.0]], dtype=wp.float32, device="cpu")
    computed = wp.zeros_like(zeros)
    applied = wp.zeros_like(zeros)

    wp.launch(
        sync_torque_telemetry,
        dim=zeros.shape,
        inputs=[
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            effort_limit,
            implicit,
            effort,
            computed_source,
            user_to_backend,
            has_ordering,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    if has_ordering:
        np.testing.assert_allclose(computed.numpy(), [[30.0, 100.0, 20.0]])
        np.testing.assert_allclose(applied.numpy(), [[300.0, 100.0, 200.0]])
    else:
        np.testing.assert_allclose(computed.numpy(), [[10.0, 200.0, 30.0]])
        np.testing.assert_allclose(applied.numpy(), [[100.0, 200.0, 300.0]])
