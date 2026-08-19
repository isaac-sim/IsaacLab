# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PD actuator equivalence tests on ANYmal-C (floating-base quadruped) — PhysX backend.

Compares IsaacLab-native actuators against Newton-native actuators (created
from the same Lab configs via USD authoring, stepped via
:class:`PhysxActuatorWrapper`) on the PhysX physics backend.  Both paths
must produce identical joint trajectories within tolerance.

Using ANYmal-C — a 12-DOF quadruped on a floating base — exercises the
full Lab-to-Newton config translation pipeline on a real-world robot.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import functools
import os
import unittest
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation
from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils.actuator_equivalence import (
    CARTPOLE_EXPLICIT_ACTUATORS,
    DC_MOTOR_ACTUATORS,
    DELAYED_PD_ACTUATORS,
    IDEAL_PD_ACTUATORS,
    IMPLICIT_ONLY_ACTUATORS,
    MIXED_WITH_IMPLICIT_ACTUATORS,
    ActuatorStateResetBase,
    EquivalenceAssertionsMixin,
    MockEnv,
    build_dr_term,
    make_dummy_lstm_checkpoint,
    make_dummy_mlp_checkpoint,
)
from isaaclab.test.utils.articulation_ordering import assert_articulation_ordering_trace_matches

from isaaclab_assets import ANYMAL_C_CFG
from isaaclab_assets.robots.spot import joint_parameter_lookup as SPOT_KNEE_LOOKUP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ENVS = 2
NUM_STEPS = 10
DT = 1.0 / 120.0
TARGET_OFFSET = 0.1  # [rad] added to initial joint positions
_ANYMAL_C_PHYSX_JOINT_NAMES = (
    "LF_HAA",
    "LH_HAA",
    "RF_HAA",
    "RH_HAA",
    "LF_HFE",
    "LH_HFE",
    "RF_HFE",
    "RH_HFE",
    "LF_KFE",
    "LH_KFE",
    "RF_KFE",
    "RH_KFE",
)


def test_prepare_native_actuators_does_not_zero_solver_gains(monkeypatch):
    """Leave solver gains untouched until collection construction resolves actuator defaults."""
    from isaaclab_physx.assets.articulation import actuator_control

    from isaaclab.actuators.newton import NewtonActuatorAdapter, PhysxActuatorWrapper

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


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_simulation(
    actuators: dict,
    use_newton_actuators: bool,
    *,
    num_steps: int = NUM_STEPS,
    feedforward: float | None = None,
    joint_ordering: tuple[str, ...] | None = None,
    permutation_sensitive_commands: bool = False,
    capture_first_compute: bool = False,
) -> dict:
    """Run ANYmal-C on PhysX and return recorded trajectories + telemetry.

    Always records public joint state/telemetry and the Newton adapter outputs.

    Args:
        actuators: Actuator configuration replacing ANYmal-C defaults.
        use_newton_actuators: Whether to use the Newton actuator fast path.
        num_steps: Number of simulation steps to record.
        feedforward: Optional constant effort target for every joint.
        joint_ordering: Optional explicit public joint-name order.
        permutation_sensitive_commands: Whether to command distinct position, velocity, and effort values by
            physical joint name.
        capture_first_compute: Whether to invoke the first actuator computation inside an outer CUDA capture.

    Returns:
        Recorded joint-name metadata, commands, public trajectories and torque telemetry, and adapter effort traces.
    """
    sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=use_newton_actuators)
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None
        for i in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
        art_cfg = ANYMAL_C_CFG.replace(
            actuators=actuators,
            prim_path="/World/Env_[^/]*/Robot",
            joint_ordering=joint_ordering,
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        joint_names = tuple(articulation.joint_names)
        backend_joint_names = tuple(articulation.backend_joint_names)
        installed_ordering = articulation.joint_ordering
        joint_ordering_state = (
            None
            if installed_ordering is None
            else {
                "user_names": joint_names,
                "backend_names": backend_joint_names,
                "user_to_backend_indices": installed_ordering.user_to_backend_indices,
                "backend_to_user_indices": installed_ordering.backend_to_user_indices,
            }
        )
        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        if permutation_sensitive_commands:
            scale_by_name = {name: index + 1 for index, name in enumerate(backend_joint_names)}
            joint_scale = torch.tensor(
                [scale_by_name[name] for name in joint_names],
                device=articulation.device,
                dtype=init_pos.dtype,
            ).unsqueeze(0)
            joint_scale = joint_scale.expand_as(init_pos)
            target_pos = init_pos + 0.01 * joint_scale
            target_vel = 0.001 * joint_scale
            effort_target = 0.1 * joint_scale
        else:
            target_pos = init_pos + TARGET_OFFSET
            target_vel = torch.zeros_like(init_pos)
            effort_target = None if feedforward is None else torch.full_like(init_pos, feedforward)

        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)
        if effort_target is not None:
            articulation.set_joint_effort_target_index(target=effort_target)

        recorded_pos, recorded_vel = [], []
        recorded_computed_effort, recorded_applied_effort = [], []
        recorded_adapter_applied = []
        if capture_first_compute:
            with wp.ScopedCapture(device=articulation.device, force_module_load=True):
                articulation.actuators.compute(DT)
        for _ in range(num_steps):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)
            recorded_pos.append(wp.to_torch(articulation.data.joint_pos).clone())
            recorded_vel.append(wp.to_torch(articulation.data.joint_vel).clone())
            recorded_computed_effort.append(articulation.actuators.computed_effort.torch.clone())
            recorded_applied_effort.append(articulation.actuators.applied_effort.torch.clone())
            if use_newton_actuators:
                recorded_adapter_applied.append(wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d).clone())
        native_actuator_graph_count = len(getattr(articulation._actuator_control, "_native_actuator_graphs", ()) or ())

    return {
        "joint_names": joint_names,
        "backend_joint_names": backend_joint_names,
        "joint_ordering": joint_ordering_state,
        "adapter_joint_names": joint_names,
        "joint_pos": recorded_pos,
        "joint_vel": recorded_vel,
        "computed_effort": recorded_computed_effort,
        "applied_effort": recorded_applied_effort,
        "adapter_applied_effort": recorded_adapter_applied,
        "target_pos": target_pos.clone(),
        "target_vel": target_vel.clone(),
        "effort_target": None if effort_target is None else effort_target.clone(),
        "native_actuator_graph_count": native_actuator_graph_count,
    }


def test_graphable_newton_actuators_capture_ping_pong_graphs() -> None:
    result = _run_simulation(DELAYED_PD_ACTUATORS, use_newton_actuators=True, num_steps=2)

    assert result["native_actuator_graph_count"] == 2


def test_newton_actuator_graph_capture_failure_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingCapture:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("capture unavailable")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(wp, "ScopedCapture", FailingCapture)

    result = _run_simulation(
        DC_MOTOR_ACTUATORS,
        use_newton_actuators=True,
        num_steps=2,
        feedforward=1.0,
    )

    assert result["native_actuator_graph_count"] == 0
    assert len(result["joint_pos"]) == 2
    assert all(torch.isfinite(joint_pos).all() for joint_pos in result["joint_pos"])
    assert all(torch.any(effort != 0.0) for effort in result["applied_effort"])
    assert all(torch.any(effort != 0.0) for effort in result["adapter_applied_effort"])


def test_stateful_newton_actuators_reject_outer_cuda_capture() -> None:
    with pytest.raises(RuntimeError, match="stateful Newton actuators cannot run inside an outer CUDA graph capture"):
        _run_simulation(
            DELAYED_PD_ACTUATORS,
            use_newton_actuators=True,
            num_steps=0,
            capture_first_compute=True,
        )


def test_newton_actuator_rollout_matches_reversed_joint_ordering() -> None:
    """Match PhysX Newton-actuator traces under reversed public joint ordering."""
    identity_result = _run_simulation(
        IDEAL_PD_ACTUATORS,
        use_newton_actuators=True,
        permutation_sensitive_commands=True,
    )
    requested_joint_names = tuple(reversed(identity_result["joint_names"]))
    reversed_result = _run_simulation(
        IDEAL_PD_ACTUATORS,
        use_newton_actuators=True,
        joint_ordering=requested_joint_names,
        permutation_sensitive_commands=True,
    )

    assert_articulation_ordering_trace_matches(identity_result, reversed_result, requested_joint_names)


def _assert_newton_actuator_uses_current_joint_state(
    joint_ordering: tuple[str, ...] | None, *, num_steps: int = NUM_STEPS
) -> None:
    """Check that ``applied_effort`` always matches the IdealPD formula on *this* step's true state.

    Ground truth is read every step via ``root_view.get_dof_positions()``/``get_dof_velocities()`` --
    the raw PhysX view, bypassing :class:`ArticulationData`'s cached ``joint_pos``/``joint_vel`` shadow
    entirely -- so the read itself cannot refresh (and thereby mask staleness in) the shadow under test.

    Args:
        joint_ordering: Optional explicit public joint-name order to install on the articulation.
        num_steps: Number of simulation steps to check.
    """
    kp, kd, effort_limit = 40.0, 5.0, 80.0
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            stiffness=kp,
            damping=kd,
            actuator_effort_limit=effort_limit,
        ),
    }
    sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=True)
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None
        for i in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
        art_cfg = ANYMAL_C_CFG.replace(
            actuators=actuators,
            prim_path="/World/Env_[^/]*/Robot",
            joint_ordering=joint_ordering,
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        ordering = articulation.joint_ordering
        if ordering is not None:
            user_to_backend = torch.tensor(
                ordering.user_to_backend_indices, dtype=torch.long, device=articulation.device
            )

            def to_user_order(raw_backend: wp.array) -> torch.Tensor:
                return wp.to_torch(raw_backend).index_select(1, user_to_backend)

        else:

            def to_user_order(raw_backend: wp.array) -> torch.Tensor:
                return wp.to_torch(raw_backend)

        init_pos = to_user_order(articulation.root_view.get_dof_positions()).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)
        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)

        for step in range(num_steps):
            # Ground truth for *this* step, independent of ArticulationData's joint_pos/joint_vel shadow.
            true_pos = to_user_order(articulation.root_view.get_dof_positions()).clone()
            true_vel = to_user_order(articulation.root_view.get_dof_velocities()).clone()

            articulation.write_data_to_sim()
            applied = articulation.actuators.applied_effort.torch.clone()

            expected = torch.clamp(kp * (target_pos - true_pos) - kd * true_vel, -effort_limit, effort_limit)
            torch.testing.assert_close(
                applied,
                expected,
                atol=1e-3,
                rtol=1e-3,
                msg=(
                    f"applied_effort at step {step} does not match the IdealPD formula evaluated on this"
                    " step's true PhysX joint state -- the Newton actuator likely used a stale"
                    " joint_pos/joint_vel shadow"
                ),
            )

            sim.step()
            articulation.update(DT)


def test_newton_actuator_identity_ordering_uses_current_joint_state() -> None:
    """Sanity check: with identity joint ordering, ``applied_effort`` always reflects this step's state."""
    _assert_newton_actuator_uses_current_joint_state(None)


def test_newton_actuator_reversed_ordering_uses_current_joint_state() -> None:
    """Regression test: non-identity joint ordering must preserve current-state torque evaluation.

    The adapter must resolve joint state in articulation order before evaluating the actuator model,
    regardless of the backend view ordering.
    """
    reversed_joint_names = tuple(reversed(_ANYMAL_C_PHYSX_JOINT_NAMES))
    _assert_newton_actuator_uses_current_joint_state(reversed_joint_names)


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class _EquivalenceTestBase(EquivalenceAssertionsMixin, unittest.TestCase):
    """Base for Lab-vs-Newton equivalence tests on the PhysX backend.

    Subclasses set ``actuators`` to the config under test.  ``setUpClass``
    runs the simulation with both ``use_newton_actuators=False`` (Lab path)
    and ``True`` (Newton via PhysxActuatorWrapper) and stores the results.
    The ``test_*_match`` oracles come from :class:`EquivalenceAssertionsMixin`.
    """

    __test__ = False
    actuators: dict = {}
    feedforward: float | None = None

    @classmethod
    def setUpClass(cls):
        cls.lab_result = _run_simulation(
            cls.actuators,
            use_newton_actuators=False,
            feedforward=cls.feedforward,
        )
        cls.newton_result = _run_simulation(
            cls.actuators,
            use_newton_actuators=True,
            feedforward=cls.feedforward,
        )


# ---------------------------------------------------------------------------
# Equivalence tests with different actuator types
# ---------------------------------------------------------------------------


class TestIdealPDEquivalence(_EquivalenceTestBase):
    """IdealPDActuator on all 12 joints: Lab vs Newton (PhysX backend)."""

    __test__ = True
    actuators = IDEAL_PD_ACTUATORS


class TestDCMotorEquivalence(_EquivalenceTestBase):
    """DCMotor actuator on all 12 joints: Lab vs Newton (PhysX backend)."""

    __test__ = True
    actuators = DC_MOTOR_ACTUATORS


class TestDelayedPDEquivalence(_EquivalenceTestBase):
    """DelayedPDActuator on all 12 joints: Lab vs Newton (PhysX).

    Verifies that actuator command delays are correctly authored and
    produce matching trajectories on the PhysX backend.
    """

    __test__ = True
    actuators = DELAYED_PD_ACTUATORS


class TestMixedWithImplicitEquivalence(_EquivalenceTestBase):
    """Implicit HAA + IdealPD HFE + DCMotor KFE: Lab vs Newton (PhysX).

    Verifies that implicit actuators (handled by PhysX joint drives)
    coexist correctly with explicit Newton actuators via PhysxActuatorWrapper.
    """

    __test__ = True
    actuators = MIXED_WITH_IMPLICIT_ACTUATORS


# ---------------------------------------------------------------------------
# Implicit + non-zero feedforward effort target on PhysX
# ---------------------------------------------------------------------------


class TestImplicitWithFeedforwardEquivalencePhysx(_EquivalenceTestBase):
    """Implicit-only actuators with a non-zero feedforward effort target on PhysX."""

    __test__ = True
    actuators = IMPLICIT_ONLY_ACTUATORS
    feedforward = 5.0


# ---------------------------------------------------------------------------
# Heterogeneous multi-articulation (ANYmal floating-base + Cartpole fixed-base)
# ---------------------------------------------------------------------------


def _run_anymal_and_cartpole(use_newton_actuators: bool, *, num_steps: int = NUM_STEPS) -> dict:
    """Spawn ANYmal-C + Cartpole per env on PhysX (different DOF counts, base types)."""
    from isaaclab_assets import CARTPOLE_CFG  # noqa: PLC0415

    sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=use_newton_actuators)
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None

        for i in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 6.0, 0, 0))

        anymal_cfg = ANYMAL_C_CFG.replace(actuators=IDEAL_PD_ACTUATORS, prim_path="/World/Env_[^/]*/Anymal")
        cartpole_cfg = CARTPOLE_CFG.replace(
            actuators=CARTPOLE_EXPLICIT_ACTUATORS,
            prim_path="/World/Env_[^/]*/Cartpole",
        )
        cartpole_cfg.init_state = cartpole_cfg.init_state.replace(pos=(0.0, 3.0, 2.0))

        anymal = Articulation(anymal_cfg)
        cartpole = Articulation(cartpole_cfg)
        sim.reset()
        assert anymal.is_initialized and cartpole.is_initialized

        init_anymal = wp.to_torch(anymal.data.joint_pos).clone()
        init_cartpole = wp.to_torch(cartpole.data.joint_pos).clone()
        anymal.set_joint_position_target_index(target=init_anymal + TARGET_OFFSET)
        anymal.set_joint_velocity_target_index(target=torch.zeros_like(init_anymal))
        cartpole.set_joint_position_target_index(target=init_cartpole + TARGET_OFFSET)
        cartpole.set_joint_velocity_target_index(target=torch.zeros_like(init_cartpole))

        pos_anymal, pos_cartpole = [], []
        for _ in range(num_steps):
            anymal.write_data_to_sim()
            cartpole.write_data_to_sim()
            sim.step()
            anymal.update(DT)
            cartpole.update(DT)
            pos_anymal.append(wp.to_torch(anymal.data.joint_pos).clone())
            pos_cartpole.append(wp.to_torch(cartpole.data.joint_pos).clone())

    return {"joint_pos_anymal": pos_anymal, "joint_pos_cartpole": pos_cartpole}


class TestHeterogeneousMultiArticulationPhysx(unittest.TestCase):
    """Two structurally-different articulations (ANYmal floating + Cartpole fixed) on PhysX.

    Each PhysX articulation owns its own :class:`PhysxActuatorWrapper`
    and per-art :class:`NewtonActuatorAdapter`. Heterogeneous DOF counts
    (12 vs 2) and base types (floating vs fixed) verify the
    per-articulation authoring + adapter construction works for varied
    structures. Equivalence against the Lab actuator path is the
    meaningful end-to-end check.
    """

    @classmethod
    def setUpClass(cls):
        cls.lab_result = _run_anymal_and_cartpole(use_newton_actuators=False)
        cls.newton_result = _run_anymal_and_cartpole(use_newton_actuators=True)

    def test_anymal_matches_lab(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["joint_pos_anymal"], self.newton_result["joint_pos_anymal"])
        ):
            torch.testing.assert_close(
                newton,
                lab,
                atol=2e-3,
                rtol=1e-3,
                msg=f"ANYmal joint_pos diverged from Lab path at step {step_i}",
            )

    def test_cartpole_matches_lab(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["joint_pos_cartpole"], self.newton_result["joint_pos_cartpole"])
        ):
            torch.testing.assert_close(
                newton,
                lab,
                atol=2e-3,
                rtol=1e-3,
                msg=f"Cartpole joint_pos diverged from Lab path at step {step_i}",
            )


# ---------------------------------------------------------------------------
# Domain randomization via events.py — PhysX backend
# ---------------------------------------------------------------------------


class TestRandomizeActuatorGainsViaEventsPhysx(unittest.TestCase):
    """End-to-end DR test for the PhysX backend.

    Drives ``randomize_actuator_gains`` (events.py) and verifies the new
    kp/kd values reach the controllers of the articulation's Newton
    actuators — exercising the full path: events → the actuator adapter →
    write_stiffness/damping → propagation to controllers. The assertions
    read the controllers back via the public
    ``read_group_parameter``.

    The native-controller tests use degenerate ranges for exact expected values.
    The implicit-storage regression instead seeds the generator and uses
    non-degenerate ranges to verify one sampled payload reaches every storage.
    """

    def test_implicit_storage_reuses_randomized_payload(self):
        """Keep actuator-owned and implicit-solver gains identical after randomization."""
        sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=False)
        with build_simulation_context(
            device="cuda:0",
            gravity_enabled=True,
            add_ground_plane=True,
            sim_cfg=sim_cfg,
        ) as sim:
            sim._app_control_on_stop_handle = None
            for i in range(NUM_ENVS):
                sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
            art_cfg = ANYMAL_C_CFG.replace(
                actuators=IMPLICIT_ONLY_ACTUATORS,
                prim_path="/World/Env_.*/Robot",
            )
            anymal = Articulation(art_cfg)
            sim.reset()

            actuator = anymal.actuators["legs"]
            stiffness_before = actuator.stiffness.clone()
            damping_before = actuator.damping.clone()
            env = MockEnv({"robot": anymal}, NUM_ENVS, anymal.device)
            term, asset_cfg = build_dr_term(env, "robot")
            env_ids = torch.tensor([0], device=anymal.device, dtype=torch.long)
            torch.manual_seed(12345)

            term(
                env,
                env_ids=env_ids,
                asset_cfg=asset_cfg,
                stiffness_distribution_params=(25.0, 75.0),
                damping_distribution_params=(1.0, 9.0),
                operation="abs",
                distribution="uniform",
            )

            randomized_stiffness = actuator.stiffness[env_ids]
            randomized_damping = actuator.damping[env_ids]
            self.assertGreater(torch.unique(randomized_stiffness).numel(), 1)
            self.assertGreater(torch.unique(randomized_damping).numel(), 1)
            torch.testing.assert_close(randomized_stiffness, anymal.data.joint_stiffness.torch[env_ids])
            torch.testing.assert_close(randomized_damping, anymal.data.joint_damping.torch[env_ids])
            torch.testing.assert_close(actuator.stiffness[1:], stiffness_before[1:])
            torch.testing.assert_close(actuator.damping[1:], damping_before[1:])

    def test_single_articulation(self):
        sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=True)
        with build_simulation_context(
            device="cuda:0",
            gravity_enabled=True,
            add_ground_plane=True,
            sim_cfg=sim_cfg,
        ) as sim:
            sim._app_control_on_stop_handle = None
            for i in range(NUM_ENVS):
                sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
            art_cfg = ANYMAL_C_CFG.replace(
                actuators=IDEAL_PD_ACTUATORS,
                prim_path="/World/Env_[^/]*/Robot",
            )
            anymal = Articulation(art_cfg)
            sim.reset()

            adapter = anymal.newton_actuator_adapter
            self.assertIsNotNone(adapter, "PhysX per-articulation adapter should exist")
            read = functools.partial(read_group_parameter, anymal.actuators)
            n = anymal.num_joints
            kp_before = read("legs", "controller", "kp").clone()
            kd_before = read("legs", "controller", "kd").clone()

            env = MockEnv({"robot": anymal}, NUM_ENVS, anymal.device)
            term, asset_cfg = build_dr_term(env, "robot")
            env_ids = torch.tensor([0], device=anymal.device, dtype=torch.long)

            term(
                env,
                env_ids=env_ids,
                asset_cfg=asset_cfg,
                stiffness_distribution_params=(100.0, 100.0),
                damping_distribution_params=(5.0, 5.0),
                operation="abs",
                distribution="uniform",
            )

            # Named native-group reads project the controller values immediately.
            torch.testing.assert_close(
                read("legs", "controller", "kp")[0], torch.full((n,), 100.0, device=anymal.device)
            )
            torch.testing.assert_close(read("legs", "controller", "kd")[0], torch.full((n,), 5.0, device=anymal.device))
            # Other envs untouched.
            for env_idx in range(1, NUM_ENVS):
                torch.testing.assert_close(read("legs", "controller", "kp")[env_idx], kp_before[env_idx])
                torch.testing.assert_close(read("legs", "controller", "kd")[env_idx], kd_before[env_idx])

    def test_two_articulations(self):
        from isaaclab_assets import CARTPOLE_CFG  # noqa: PLC0415

        sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=True)
        with build_simulation_context(
            device="cuda:0",
            gravity_enabled=True,
            add_ground_plane=True,
            sim_cfg=sim_cfg,
        ) as sim:
            sim._app_control_on_stop_handle = None
            for i in range(NUM_ENVS):
                sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 6.0, 0, 0))

            anymal_cfg = ANYMAL_C_CFG.replace(actuators=IDEAL_PD_ACTUATORS, prim_path="/World/Env_[^/]*/Anymal")
            cartpole_cfg = CARTPOLE_CFG.replace(
                actuators=CARTPOLE_EXPLICIT_ACTUATORS,
                prim_path="/World/Env_[^/]*/Cartpole",
            )
            cartpole_cfg.init_state = cartpole_cfg.init_state.replace(pos=(0.0, 3.0, 2.0))
            anymal = Articulation(anymal_cfg)
            cartpole = Articulation(cartpole_cfg)
            sim.reset()

            # On PhysX each articulation owns its own adapter — they are distinct objects.
            anymal_adapter = anymal.newton_actuator_adapter
            cartpole_adapter = cartpole.newton_actuator_adapter
            self.assertIsNotNone(anymal_adapter)
            self.assertIsNotNone(cartpole_adapter)
            self.assertIsNot(anymal_adapter, cartpole_adapter)

            anymal_read = functools.partial(read_group_parameter, anymal.actuators)
            cartpole_read = functools.partial(read_group_parameter, cartpole.actuators)
            n_cp = cartpole.num_joints
            anymal_kp_before = anymal_read("legs", "controller", "kp").clone()
            anymal_kd_before = anymal_read("legs", "controller", "kd").clone()
            cp_kp_before = cartpole_read("all_joints", "controller", "kp").clone()
            cp_kd_before = cartpole_read("all_joints", "controller", "kd").clone()

            env = MockEnv({"anymal": anymal, "cartpole": cartpole}, NUM_ENVS, anymal.device)
            term, asset_cfg = build_dr_term(env, "cartpole")
            env_ids = torch.tensor([0], device=anymal.device, dtype=torch.long)

            term(
                env,
                env_ids=env_ids,
                asset_cfg=asset_cfg,
                stiffness_distribution_params=(100.0, 100.0),
                damping_distribution_params=(5.0, 5.0),
                operation="abs",
                distribution="uniform",
            )

            cp_kp_after = cartpole_read("all_joints", "controller", "kp")
            cp_kd_after = cartpole_read("all_joints", "controller", "kd")
            torch.testing.assert_close(cp_kp_after[0], torch.full((n_cp,), 100.0, device=anymal.device))
            torch.testing.assert_close(cp_kd_after[0], torch.full((n_cp,), 5.0, device=anymal.device))
            # Cartpole's other envs are untouched (env_ids=[0] only).
            for env_idx in range(1, NUM_ENVS):
                torch.testing.assert_close(cp_kp_after[env_idx], cp_kp_before[env_idx])
                torch.testing.assert_close(cp_kd_after[env_idx], cp_kd_before[env_idx])

            # ANYmal's controllers are fully untouched — DR was scoped to cartpole.
            torch.testing.assert_close(anymal_read("legs", "controller", "kp"), anymal_kp_before)
            torch.testing.assert_close(anymal_read("legs", "controller", "kd"), anymal_kd_before)


# ---------------------------------------------------------------------------
# Per-env reset: actuator state isolation
# ---------------------------------------------------------------------------


class TestActuatorStateReset(ActuatorStateResetBase, unittest.TestCase):
    """Per-env actuator state reset isolation on the PhysX backend.

    The scenario and assertions live in :class:`ActuatorStateResetBase`;
    this subclass provides the PhysX sim config and the per-articulation
    adapter (``articulation.newton_actuator_adapter``).
    """

    def _make_sim_cfg(self, use_newton_actuators: bool) -> SimulationCfg:
        return SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=use_newton_actuators)

    def _make_articulation(self) -> Articulation:
        return Articulation(ANYMAL_C_CFG.replace(actuators=DELAYED_PD_ACTUATORS, prim_path="/World/Env_.*/Robot"))

    def _get_adapter(self, articulation):
        return articulation.newton_actuator_adapter


# ---------------------------------------------------------------------------
# RemotizedPD equivalence: PD + delay + position-based clamping lookup table
# ---------------------------------------------------------------------------


class TestRemotizedPDEquivalence(_EquivalenceTestBase):
    """RemotizedPD (PD + delay + position-based clamping): Lab vs Newton (PhysX).

    Uses the Spot knee lookup table on ANYmal's KFE joints with IdealPD
    on HAA and HFE.
    """

    __test__ = True

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

        cls.actuators = {
            "hips": IdealPDActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE"],
                stiffness=40.0,
                damping=5.0,
                actuator_effort_limit=80.0,
            ),
            "knees": RemotizedPDActuatorCfg(
                joint_names_expr=[".*KFE"],
                stiffness=60.0,
                damping=1.5,
                actuator_effort_limit=80.0,
                max_delay=3,
                joint_parameter_lookup=SPOT_KNEE_LOOKUP,
            ),
        }
        super().setUpClass()


# ---------------------------------------------------------------------------
# Neural network actuator authoring: MLP and LSTM
# ---------------------------------------------------------------------------


class TestNeuralMLPFunctional(unittest.TestCase):
    """Verify ActuatorNetMLPCfg runs on PhysX with Newton actuators."""

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg  # noqa: PLC0415

        cls.mlp_path = make_dummy_mlp_checkpoint()
        cls.result = _run_simulation(
            {
                "mlp_legs": ActuatorNetMLPCfg(
                    joint_names_expr=[".*HAA"],
                    network_file=cls.mlp_path,
                    saturation_effort=120.0,
                    actuator_effort_limit=80.0,
                    actuator_velocity_limit=7.5,
                    pos_scale=-1.0,
                    vel_scale=1.0,
                    torque_scale=1.0,
                    input_order="pos_vel",
                    input_idx=[0, 1, 2],
                ),
                "pd_legs": IdealPDActuatorCfg(
                    joint_names_expr=[".*HFE", ".*KFE"],
                    stiffness=40.0,
                    damping=5.0,
                    actuator_effort_limit=80.0,
                ),
            },
            use_newton_actuators=True,
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.mlp_path)

    def test_positions_finite(self):
        for step_i, pos in enumerate(self.result["joint_pos"]):
            self.assertTrue(
                torch.isfinite(pos).all(),
                f"Non-finite positions at step {step_i}",
            )


class TestNeuralLSTMFunctional(unittest.TestCase):
    """Verify ActuatorNetLSTMCfg runs on PhysX with Newton actuators."""

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg  # noqa: PLC0415

        cls.lstm_path = make_dummy_lstm_checkpoint()
        cls.result = _run_simulation(
            {
                "lstm_legs": ActuatorNetLSTMCfg(
                    joint_names_expr=[".*HAA"],
                    network_file=cls.lstm_path,
                    saturation_effort=120.0,
                    actuator_effort_limit=80.0,
                    actuator_velocity_limit=7.5,
                ),
                "pd_legs": IdealPDActuatorCfg(
                    joint_names_expr=[".*HFE", ".*KFE"],
                    stiffness=40.0,
                    damping=5.0,
                    actuator_effort_limit=80.0,
                ),
            },
            use_newton_actuators=True,
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.lstm_path)

    def test_positions_finite(self):
        for step_i, pos in enumerate(self.result["joint_pos"]):
            self.assertTrue(
                torch.isfinite(pos).all(),
                f"Non-finite positions at step {step_i}",
            )


if __name__ == "__main__":
    unittest.main()
