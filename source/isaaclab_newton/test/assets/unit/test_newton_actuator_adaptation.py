# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Newton actuator adaptation: target modes resolved on the model builder, native-group selection, and telemetry.

These run on a model builder and on Warp arrays directly; none of them steps a simulation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import JointTargetMode, JointType, ModelBuilder
from newton.solvers import SolverMuJoCo

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.actuators.newton.kernels import sync_torque_telemetry
from isaaclab.assets import ArticulationCfg

pytestmark = pytest.mark.unit


class CustomDrive(ImplicitActuator):
    """Implicit actuator with a class name that does not encode its execution type."""


def _make_target_mode_builder(
    monkeypatch, joint_names: list[str], modes: list[JointTargetMode], stiffness: list[float], damping: list[float]
) -> ModelBuilder:
    """Build a zero-gain articulated model builder for target-mode tests."""
    sim = object.__new__(sim_utils.SimulationContext)
    sim.cfg = SimpleNamespace(physics=NewtonCfg())
    sim._backend_registry = []
    monkeypatch.setattr(sim_utils.SimulationContext, "instance", lambda: sim)
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = -1
    joint_ids = []
    for joint_name in joint_names:
        link = builder.add_link(mass=1.0, inertia=inertia, label=f"/World/Env_0/Robot/{joint_name}_link")
        joint_ids.append(
            builder.add_joint_revolute(
                parent,
                link,
                target_ke=0.0,
                target_kd=0.0,
                label=f"/World/Env_0/Robot/{joint_name}",
            )
        )
        parent = link
    builder.add_articulation(joint_ids, label="/World/Env_0/Robot")
    builder.articulation_label = ["/World/Env_0/Robot"]
    builder.joint_target_mode = [int(mode) for mode in modes]
    builder.joint_target_ke = stiffness
    builder.joint_target_kd = damping
    return builder


@pytest.mark.parametrize(
    ("actuator_cfg", "expected_native_groups"),
    [
        (ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0), set()),
        (IdealPDActuatorCfg(joint_names_expr=["joint"], stiffness=None, damping=None), {"explicit"}),
    ],
    ids=["implicit", "explicit"],
)
def test_prepare_native_actuators_activates_only_explicit_groups(monkeypatch, actuator_cfg, expected_native_groups):
    """Keep implicit-only articulations on the solver-drive path and leave solver gains untouched.

    Explicit groups activate the Newton-actuator path without writing gains; collection construction resolves
    the actuator defaults later.
    """
    activation_calls = []
    gain_writes = []
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        device="cpu",
        find_joints=lambda _: ([0], ["joint"]),
        write_joint_stiffness_to_sim_index=lambda **_: gain_writes.append("stiffness"),
        write_joint_damping_to_sim_index=lambda **_: gain_writes.append("damping"),
    )
    monkeypatch.setattr(SimulationManager, "activate_newton_actuator_path", lambda: activation_calls.append(True))

    control = NewtonActuatorControl(articulation)
    group_name = "explicit" if expected_native_groups else "implicit"
    native_groups = control.prepare_native_actuators(collection=None, actuator_cfgs={group_name: actuator_cfg})

    assert native_groups == expected_native_groups
    assert gain_writes == []
    if expected_native_groups:
        assert activation_calls == [True]
    else:
        assert not control.native_actuator_path_active
        assert not articulation._has_newton_actuators
        assert activation_calls == []


@pytest.mark.parametrize(
    ("actuator_cfg", "expected_mode", "expected_actuator_indices"),
    [
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0),
            JointTargetMode.POSITION,
            [0, 1],
        ),
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=2.0),
            JointTargetMode.VELOCITY,
            [-2, -3],
        ),
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=2.0),
            JointTargetMode.POSITION_VELOCITY,
            [0, -2, 1, -3],
        ),
        (
            ImplicitActuatorCfg(
                class_type=f"{__name__}:CustomDrive", joint_names_expr=[".*"], stiffness=10.0, damping=2.0
            ),
            JointTargetMode.POSITION_VELOCITY,
            [0, -2, 1, -3],
        ),
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
            JointTargetMode.EFFORT,
            None,
        ),
        (
            IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=2.0),
            JointTargetMode.EFFORT,
            None,
        ),
    ],
)
def test_actuator_cfg_sets_newton_target_mode_before_solver_init(
    monkeypatch, actuator_cfg, expected_mode, expected_actuator_indices
):
    """Resolve configured modes before finalization constructs MuJoCo actuators."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": actuator_cfg},
    )
    builder = _make_target_mode_builder(
        monkeypatch, ["left_joint", "right_joint"], [JointTargetMode.NONE, JointTargetMode.NONE], [0.0, 0.0], [0.0, 0.0]
    )
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=articulation_cfg), None)
    model = builder.finalize(device="cpu")
    solver = SolverMuJoCo(model, use_mujoco_cpu=True)
    assert model.joint_target_mode.numpy().tolist() == [int(expected_mode), int(expected_mode)]
    assert (
        solver.mjc_actuator_to_newton_idx.numpy().tolist() if solver.mjc_actuator_to_newton_idx is not None else None
    ) == expected_actuator_indices


def test_actuator_cfg_matches_explicit_descendant_articulation_root(monkeypatch):
    """Match target modes against an explicitly configured descendant articulation root."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="/base",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    builder = _make_target_mode_builder(monkeypatch, ["joint"], [JointTargetMode.NONE], [0.0], [0.0])
    builder.articulation_label = ["/World/Env_0/Robot/base"]
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=articulation_cfg), None)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)]


def test_actuator_cfg_matches_clone_plan_root_expr(monkeypatch):
    """Match builder labels against the clone slot spelling clone-plan root resolution returns."""
    articulation_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    monkeypatch.setattr(
        "isaaclab_newton.assets.articulation.articulation.resolve_matching_prims_from_source",
        lambda *_args, **_kwargs: [(None, "/World/envs/env_[^/]+/Robot/base")],
    )
    builder = _make_target_mode_builder(monkeypatch, ["joint"], [JointTargetMode.NONE], [0.0], [0.0])
    builder.articulation_label = ["/World/envs/env_0/Robot/base"]
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=articulation_cfg), None)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)]


@pytest.mark.parametrize("joint_type", [JointType.FREE, JointType.FIXED])
def test_actuator_cfg_leaves_excluded_joint_types_imported(monkeypatch, joint_type):
    """Leave target modes for free and fixed joints unchanged."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    builder = _make_target_mode_builder(monkeypatch, ["joint"], [JointTargetMode.NONE], [0.0], [0.0])
    builder.joint_type[0] = joint_type
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=articulation_cfg), None)
    assert builder.joint_target_mode == [int(JointTargetMode.NONE)]


def test_actuator_cfg_uses_imported_gain_for_none_stiffness(monkeypatch):
    """Retain the imported stiffness when an implicit actuator config leaves it unset."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=0.0)},
    )
    builder = _make_target_mode_builder(monkeypatch, ["joint"], [JointTargetMode.EFFORT], [10.0], [0.0])
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=articulation_cfg), None)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)]


def test_actuator_cfg_leaves_unconfigured_newton_target_modes_imported(monkeypatch):
    """Leave target modes for DOFs outside an actuator group unchanged."""
    subset_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={
            "shoulder": ImplicitActuatorCfg(joint_names_expr=["left_shoulder"], stiffness=10.0, damping=0.0),
        },
    )
    builder = _make_target_mode_builder(
        monkeypatch,
        ["left_shoulder", "right_shoulder"],
        [JointTargetMode.NONE, JointTargetMode.VELOCITY],
        [0.0, 0.0],
        [0.0, 2.0],
    )
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=subset_cfg), None)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION), int(JointTargetMode.VELOCITY)]


@pytest.mark.parametrize(
    ("stiffness", "damping", "expected_modes"),
    [
        ({"left_joint": 10.0}, 0.0, [JointTargetMode.POSITION, JointTargetMode.EFFORT]),
        ({"left_joint": 10.0}, {"right_joint": 2.0}, [JointTargetMode.POSITION, JointTargetMode.VELOCITY]),
    ],
)
def test_actuator_cfg_aligns_partial_dictionary_gains_by_joint_name(monkeypatch, stiffness, damping, expected_modes):
    """Resolve sparse stiffness and damping dictionaries independently by joint name."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
    )
    builder = _make_target_mode_builder(
        monkeypatch, ["left_joint", "right_joint"], [JointTargetMode.NONE, JointTargetMode.NONE], [0.0, 0.0], [0.0, 0.0]
    )
    Articulation._configure_joint_target_modes(SimpleNamespace(cfg=articulation_cfg), None)
    assert builder.joint_target_mode == [int(mode) for mode in expected_modes]


def test_sync_torque_telemetry_keeps_user_order_effort_buffers_unmapped() -> None:
    """Report torque telemetry directly from user-order actuator buffers."""
    joint_pos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    joint_modes = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_effort = wp.array(np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    user_computed_effort = wp.array(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    computed = wp.zeros_like(joint_pos)
    applied = wp.zeros_like(joint_pos)

    wp.launch(
        sync_torque_telemetry,
        dim=joint_pos.shape,
        inputs=[
            joint_pos,
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.zeros_like(joint_pos),
            wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu"),
            joint_modes,
            user_effort,
            user_computed_effort,
            user_to_backend,
            False,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    np.testing.assert_allclose(computed.numpy(), np.asarray([[10.0, 200.0, 30.0]], dtype=np.float32))
    np.testing.assert_allclose(applied.numpy(), np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32))
