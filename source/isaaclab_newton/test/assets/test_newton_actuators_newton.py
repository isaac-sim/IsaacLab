# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PD actuator equivalence tests on ANYmal-C (floating-base quadruped).

Compares IsaacLab-native actuators against Newton-native actuators (created
from the same Lab configs via USD authoring) on the Newton physics backend.
Both paths must produce identical joint trajectories within tolerance.

Using ANYmal-C — a 12-DOF quadruped on a floating base — exercises the
coordinate-vs-DOF index separation that is critical when free joints shift
the mapping between ``joint_q`` (coordinate layout) and ``joint_qd``
(DOF layout).

Each test class overrides ANYmal's default actuators with a specific Lab
config (IdealPD, DCMotor, or mixed) and verifies Lab vs Newton equivalence.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import functools
import os
import unittest

import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.actuators.newton.kernels import sync_torque_telemetry
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

NEWTON_CFG = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        njmax=500,
        nconmax=500,
        ls_iterations=20,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
    ),
    num_substeps=1,
    debug_mode=False,
    use_cuda_graph=False,
)

# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_simulation(
    actuators: dict,
    use_newton_actuators: bool,
    *,
    dt: float = DT,
    newton_cfg: NewtonCfg = NEWTON_CFG,
    num_steps: int = NUM_STEPS,
    decimation: int = 1,
    feedforward: float | None = None,
    joint_ordering: tuple[str, ...] | None = None,
    permutation_sensitive_commands: bool = False,
) -> dict:
    """Run ANYmal-C and return recorded trajectories + telemetry.

    Always records ``joint_pos``, ``joint_vel``, ``computed_effort``, and
    ``applied_effort`` so callers don't need a separate "with telemetry"
    runner. Optionally applies a constant per-DOF feedforward effort target.

    Args:
        actuators: Actuator config dict overriding ANYmal's defaults.
        use_newton_actuators: Use Newton-native actuators when ``True``.
        dt: Physics timestep [s].
        newton_cfg: Newton physics configuration.
        num_steps: Number of policy-level steps.
        decimation: Actuator steps per policy step (Newton's CUDA-graph
            d-loop is used when all-graphable; otherwise an explicit Python
            inner loop).
        feedforward: When not ``None``, set a constant per-DOF feedforward
            effort target. Used by the implicit-FF equivalence test.
        joint_ordering: Optional explicit public joint-name order.
        permutation_sensitive_commands: Whether to command distinct position, velocity, and effort values by
            physical joint name.

    Returns:
        Recorded joint-name metadata, commands, public trajectories and torque telemetry, and backend-order
        adapter effort traces.
    """
    sim_cfg = SimulationCfg(dt=dt, physics=newton_cfg, use_newton_actuators=use_newton_actuators)
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

        if use_newton_actuators and decimation > 1:
            SimulationManager.set_decimation(decimation)

        handles_dec = (
            use_newton_actuators
            and decimation > 1
            and SimulationManager._is_all_graphable()
            and SimulationManager._decimation > 1
        )

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
                "is_identity": False,
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
        for _ in range(num_steps):
            if handles_dec:
                articulation.write_data_to_sim()
                sim.step()
                articulation.update(dt * decimation)
            else:
                for _ in range(decimation):
                    articulation.write_data_to_sim()
                    sim.step()
                    articulation.update(dt)
            recorded_pos.append(wp.to_torch(articulation.data.joint_pos).clone())
            recorded_vel.append(wp.to_torch(articulation.data.joint_vel).clone())
            recorded_computed_effort.append(articulation.actuators.computed_effort.torch.clone())
            recorded_applied_effort.append(articulation.actuators.applied_effort.torch.clone())
            if use_newton_actuators:
                recorded_adapter_applied.append(wp.to_torch(articulation.data._sim_bind_joint_effort).clone())

    return {
        "joint_names": joint_names,
        "backend_joint_names": backend_joint_names,
        "joint_ordering": joint_ordering_state,
        "adapter_joint_names": backend_joint_names,
        "joint_pos": recorded_pos,
        "joint_vel": recorded_vel,
        "computed_effort": recorded_computed_effort,
        "applied_effort": recorded_applied_effort,
        "adapter_applied_effort": recorded_adapter_applied,
        "target_pos": target_pos.clone(),
        "target_vel": target_vel.clone(),
        "effort_target": None if effort_target is None else effort_target.clone(),
    }


def test_newton_actuator_rollout_matches_reversed_joint_ordering() -> None:
    """Match Newton-backend actuator traces under reversed public joint ordering."""
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

    installed_ordering = reversed_result["joint_ordering"]
    assert installed_ordering is not None
    assert not installed_ordering["is_identity"]
    assert_articulation_ordering_trace_matches(identity_result, reversed_result, requested_joint_names)


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class _EquivalenceTestBase(EquivalenceAssertionsMixin, unittest.TestCase):
    """Base for Lab-vs-Newton equivalence tests.

    Subclasses set ``actuators`` to the config under test.  ``setUpClass``
    runs the simulation with both ``use_newton_actuators=False`` (Lab path)
    and ``True`` (Newton path) and stores the results. The ``test_*_match``
    oracles come from :class:`EquivalenceAssertionsMixin`.
    """

    __test__ = False
    actuators: dict = {}
    feedforward: float | None = None
    dt: float = DT
    newton_cfg: NewtonCfg = NEWTON_CFG
    num_steps: int = NUM_STEPS
    decimation: int = 1

    @classmethod
    def setUpClass(cls):
        kwargs = dict(
            feedforward=cls.feedforward,
            dt=cls.dt,
            newton_cfg=cls.newton_cfg,
            num_steps=cls.num_steps,
            decimation=cls.decimation,
        )
        cls.lab_result = _run_simulation(cls.actuators, use_newton_actuators=False, **kwargs)
        cls.newton_result = _run_simulation(cls.actuators, use_newton_actuators=True, **kwargs)


# ---------------------------------------------------------------------------
# Equivalence tests with different actuator types
# ---------------------------------------------------------------------------


class TestIdealPDEquivalence(_EquivalenceTestBase):
    """IdealPDActuator on all 12 joints: Lab vs Newton."""

    __test__ = True
    actuators = IDEAL_PD_ACTUATORS


class TestDCMotorEquivalence(_EquivalenceTestBase):
    """DCMotor actuator on all 12 joints: Lab vs Newton."""

    __test__ = True
    actuators = DC_MOTOR_ACTUATORS


class TestMixedWithImplicitEquivalence(_EquivalenceTestBase):
    """Implicit HAA + IdealPD HFE + DCMotor KFE: Lab vs Newton.

    Verifies that implicit actuators (handled by the physics engine's
    built-in joint drives) coexist correctly with explicit Newton actuators.
    """

    __test__ = True
    actuators = MIXED_WITH_IMPLICIT_ACTUATORS


# ---------------------------------------------------------------------------
# Implicit + non-zero feedforward effort target
# ---------------------------------------------------------------------------


class TestImplicitWithFeedforwardEquivalence(_EquivalenceTestBase):
    """Implicit-only actuators with a non-zero feedforward effort target.

    Verifies that the user's FF effort lands additively on top of the
    simulator's joint-drive PD identically on both Lab and Newton paths.
    """

    __test__ = True
    actuators = IMPLICIT_ONLY_ACTUATORS
    feedforward = 2.0
    torque_atol = 0.5


# ---------------------------------------------------------------------------
# Multi-articulation Newton scene (regression test for class-attr clobber)
# ---------------------------------------------------------------------------


def _run_anymal_and_cartpole(use_newton_actuators: bool, *, num_steps: int = NUM_STEPS) -> dict:
    """Spawn ANYmal-C + Cartpole per env (different DOF counts, different base types)."""
    from isaaclab_assets import CARTPOLE_CFG  # noqa: PLC0415

    sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG, use_newton_actuators=use_newton_actuators)
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
        # Stand the cartpole well clear of the anymal.
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


class TestHeterogeneousMultiArticulationNewton(unittest.TestCase):
    """Two structurally-different articulations (ANYmal floating + Cartpole fixed) on Newton.

    Regression for the singleton-clobber bug in ``NewtonManager._adapter``
    / ``_post_actuator_callback`` — fixed by the global-adapter refactor
    + callback-list multiplexing. Heterogeneous DOF counts (12 vs 2) and
    base types (floating vs fixed) stress the global adapter's handling
    of varied actuator index patterns. Equivalence against the Lab
    actuator path is the meaningful end-to-end check: divergence on
    either robot would indicate broken stepping.
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
# Domain randomization via events.py — Newton backend
# ---------------------------------------------------------------------------


class TestRandomizeActuatorGainsViaEventsNewton(unittest.TestCase):
    """End-to-end DR test for the Newton backend.

    Drives ``randomize_actuator_gains`` and verifies that kp/kd values reach
    the controllers of the articulation's Newton actuators through
    ``write_group_parameter``; the assertions read the
    controllers back via the public ``read_group_parameter``.

    With ``operation="abs"`` and ``distribution="uniform"`` over a
    degenerate range ``(K, K)``, every randomized cell is set to exactly
    ``K`` — so the assertions are deterministic.
    """

    def test_single_articulation(self):
        sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG, use_newton_actuators=True)
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

            adapter = SimulationManager._adapter
            self.assertIsNotNone(adapter, "Newton adapter should exist with use_newton_actuators=True")
            read = functools.partial(read_group_parameter, anymal.actuators)
            n = anymal.num_joints
            # Before DR, native gain reads must return the configured values for
            # *every* env. IDEAL_PD_ACTUATORS covers all 12 joints with constant
            # gains, so every cell of both env rows must equal the configured
            # value. This is also the regression check for the env-major DOF
            # stride decoding on floating-base articulations (ANYmal-C has 6
            # free-root DOFs + 12 joints -> a per-env stride of 18 vs.
            # ``num_joints == 12``): a wrong stride corrupts every env past the
            # first.
            legs_stiffness_before = read("legs", "controller", "kp").clone()
            legs_damping_before = read("legs", "controller", "kd").clone()
            torch.testing.assert_close(legs_stiffness_before, torch.full((NUM_ENVS, n), 40.0, device=anymal.device))
            torch.testing.assert_close(legs_damping_before, torch.full((NUM_ENVS, n), 5.0, device=anymal.device))

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
                torch.testing.assert_close(read("legs", "controller", "kp")[env_idx], legs_stiffness_before[env_idx])
                torch.testing.assert_close(read("legs", "controller", "kd")[env_idx], legs_damping_before[env_idx])

    def test_two_articulations(self):
        from isaaclab_assets import CARTPOLE_CFG  # noqa: PLC0415

        sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG, use_newton_actuators=True)
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

            self.assertIsNotNone(SimulationManager._adapter)

            anymal_read = functools.partial(read_group_parameter, anymal.actuators)
            cartpole_read = functools.partial(read_group_parameter, cartpole.actuators)
            anymal_stiffness_before = anymal_read("legs", "controller", "kp").clone()
            anymal_damping_before = anymal_read("legs", "controller", "kd").clone()
            cartpole_stiffness_before = cartpole_read("all_joints", "controller", "kp").clone()
            cartpole_damping_before = cartpole_read("all_joints", "controller", "kd").clone()

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

            n_cp = cartpole.num_joints
            torch.testing.assert_close(
                cartpole_read("all_joints", "controller", "kp")[0], torch.full((n_cp,), 100.0, device=anymal.device)
            )
            torch.testing.assert_close(
                cartpole_read("all_joints", "controller", "kd")[0], torch.full((n_cp,), 5.0, device=anymal.device)
            )

            # ANYmal is untouched (DR was scoped to cartpole).
            torch.testing.assert_close(anymal_read("legs", "controller", "kp"), anymal_stiffness_before)
            torch.testing.assert_close(anymal_read("legs", "controller", "kd"), anymal_damping_before)

            # Cartpole's other envs are also untouched (env_ids=[0] only).
            for env_idx in range(1, NUM_ENVS):
                torch.testing.assert_close(
                    cartpole_read("all_joints", "controller", "kp")[env_idx], cartpole_stiffness_before[env_idx]
                )
                torch.testing.assert_close(
                    cartpole_read("all_joints", "controller", "kd")[env_idx], cartpole_damping_before[env_idx]
                )


# ---------------------------------------------------------------------------
# DelayedPD equivalence: PD with actuator command delay
# ---------------------------------------------------------------------------


class TestDelayedPDEquivalence(_EquivalenceTestBase):
    """DelayedPDActuator on all 12 joints: Lab vs Newton.

    Verifies that actuator command delays are correctly authored as
    ``NewtonActuatorDelayAPI`` and produce matching trajectories.
    """

    __test__ = True
    actuators = DELAYED_PD_ACTUATORS


class TestDelayedPDAuthoring(unittest.TestCase):
    """Verify DelayedPDActuatorCfg is authored with NewtonActuatorDelayAPI."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_authoring_introspection(DELAYED_PD_ACTUATORS)

    def test_has_delay(self):
        for a in self.result["actuator_info"]:
            self.assertTrue(a["has_delay"], "Delay not found on delayed PD actuator")

    def test_controller_is_pd(self):
        for a in self.result["actuator_info"]:
            self.assertEqual(a["controller_type"], "ControllerPD")


# ---------------------------------------------------------------------------
# Decimation tests: re-run equivalence with decimation > 1 + CUDA graph capture
# ---------------------------------------------------------------------------

NEWTON_CFG_DEC = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        njmax=500,
        nconmax=500,
        ls_iterations=20,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
    ),
    num_substeps=2,
    debug_mode=False,
    use_cuda_graph=True,
)


class _DecimationMixin:
    """Common knobs for decimation/CUDA-graph variants of equivalence classes."""

    __test__ = True
    dt = 1.0 / 100.0
    newton_cfg = NEWTON_CFG_DEC
    num_steps = 5
    decimation = 2


class TestDecimationDCMotor(_DecimationMixin, TestDCMotorEquivalence):
    """DCMotor — same equivalence checks, with decimation=2 + CUDA graph."""


class TestDecimationDelayedPD(_DecimationMixin, TestDelayedPDEquivalence):
    """DelayedPD — decimation=2 + CUDA graph (delay queue stepped inside the captured graph)."""


# ---------------------------------------------------------------------------
# Per-env reset: actuator state isolation
# ---------------------------------------------------------------------------


class TestActuatorStateReset(ActuatorStateResetBase, unittest.TestCase):
    """Per-env actuator state reset isolation on the Newton backend.

    The scenario and assertions live in :class:`ActuatorStateResetBase`;
    this subclass provides the Newton sim config and the model-wide adapter.
    """

    def _make_sim_cfg(self, use_newton_actuators: bool) -> SimulationCfg:
        return SimulationCfg(dt=DT, physics=NEWTON_CFG, use_newton_actuators=use_newton_actuators)

    def _make_articulation(self) -> Articulation:
        return Articulation(ANYMAL_C_CFG.replace(actuators=DELAYED_PD_ACTUATORS, prim_path="/World/Env_.*/Robot"))

    def _get_adapter(self, articulation):
        return SimulationManager._adapter


# ---------------------------------------------------------------------------
# RemotizedPD actuator: PD + delay + position-based clamping lookup table
# ---------------------------------------------------------------------------


def _remotized_pd_actuators() -> dict:
    """RemotizedPD (Spot knee lookup) on KFE with IdealPD on HAA/HFE."""
    from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

    return {
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


def _run_authoring_introspection(actuator_cfgs: dict) -> dict:
    """Instantiate Newton simulation, return Newton actuator introspection.

    Verifies that Lab configs are correctly authored to Newton USD schemas
    and that Newton creates the expected controller/clamping/delay objects.

    Returns:
        Dict with ``num_actuators``, ``actuator_info`` (list of per-actuator
        dicts), and ``joint_pos`` (recorded trajectories).
    """
    sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG, use_newton_actuators=True)

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
            actuators=actuator_cfgs,
            prim_path="/World/Env_[^/]*/Robot",
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        model = SimulationManager.get_model()

        actuator_info = []
        for act in model.actuators:
            ctrl_type = type(act.controller).__name__
            clamp_types = sorted(type(c).__name__ for c in (act.clamping or []))
            actuator_info.append(
                {
                    "controller_type": ctrl_type,
                    "clamping_types": clamp_types,
                    "has_delay": act.delay is not None,
                    "num_indices": len(act.indices),
                }
            )

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)
        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)

        recorded_pos = []
        for _ in range(NUM_STEPS):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)
            recorded_pos.append(wp.to_torch(articulation.data.joint_pos).clone())

    return {
        "num_actuators": len(model.actuators),
        "actuator_info": actuator_info,
        "joint_pos": recorded_pos,
    }


class TestRemotizedPDAuthoring(unittest.TestCase):
    """Verify RemotizedPDActuatorCfg is authored as Newton PD + delay +
    position-based clamping.

    Uses the Spot knee lookup table on ANYmal's KFE joints, with IdealPD
    on HAA and HFE joints.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _run_authoring_introspection(_remotized_pd_actuators())

    def test_num_actuators(self):
        self.assertGreaterEqual(self.result["num_actuators"], 2)

    def test_kfe_controller_is_pd(self):
        kfe_acts = [a for a in self.result["actuator_info"] if "ClampingPositionBased" in a["clamping_types"]]
        self.assertTrue(len(kfe_acts) > 0, "No actuator with position-based clamping found")
        for a in kfe_acts:
            self.assertEqual(a["controller_type"], "ControllerPD")

    def test_kfe_has_position_based_clamping(self):
        kfe_acts = [a for a in self.result["actuator_info"] if "ClampingPositionBased" in a["clamping_types"]]
        self.assertTrue(len(kfe_acts) > 0, "Position-based clamping not found")

    def test_kfe_has_delay(self):
        kfe_acts = [a for a in self.result["actuator_info"] if "ClampingPositionBased" in a["clamping_types"]]
        for a in kfe_acts:
            self.assertTrue(a["has_delay"], "Delay not found on remotized KFE actuator")


class TestRemotizedPDEquivalence(_EquivalenceTestBase):
    """RemotizedPD (PD + delay + position-based clamping): Lab vs Newton."""

    __test__ = True

    @classmethod
    def setUpClass(cls):
        cls.actuators = _remotized_pd_actuators()
        super().setUpClass()


class TestDecimationRemotizedPD(_DecimationMixin, TestRemotizedPDEquivalence):
    """RemotizedPD — decimation=2 + CUDA graph."""


# ---------------------------------------------------------------------------
# Neural network actuator authoring: MLP and LSTM
# ---------------------------------------------------------------------------


class TestNeuralMLPAuthoring(unittest.TestCase):
    """Verify ActuatorNetMLPCfg is authored as Newton NeuralMLP controller
    with DC motor clamping.
    """

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg  # noqa: PLC0415

        cls.mlp_path = make_dummy_mlp_checkpoint()
        cls.result = _run_authoring_introspection(
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
            }
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.mlp_path)

    def test_num_actuators(self):
        self.assertGreaterEqual(self.result["num_actuators"], 2)

    def test_has_neural_mlp_controller(self):
        mlp_acts = [a for a in self.result["actuator_info"] if a["controller_type"] == "ControllerNeuralMLP"]
        self.assertTrue(len(mlp_acts) > 0, "No NeuralMLP controller found")

    def test_mlp_has_dc_motor_clamping(self):
        mlp_acts = [a for a in self.result["actuator_info"] if a["controller_type"] == "ControllerNeuralMLP"]
        for a in mlp_acts:
            self.assertIn("ClampingDCMotor", a["clamping_types"])


class TestNeuralLSTMAuthoring(unittest.TestCase):
    """Verify ActuatorNetLSTMCfg is authored as Newton NeuralLSTM controller
    with DC motor clamping.
    """

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg  # noqa: PLC0415

        cls.lstm_path = make_dummy_lstm_checkpoint()
        cls.result = _run_authoring_introspection(
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
            }
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.lstm_path)

    def test_num_actuators(self):
        self.assertGreaterEqual(self.result["num_actuators"], 2)

    def test_has_neural_lstm_controller(self):
        lstm_acts = [a for a in self.result["actuator_info"] if a["controller_type"] == "ControllerNeuralLSTM"]
        self.assertTrue(len(lstm_acts) > 0, "No NeuralLSTM controller found")

    def test_lstm_has_dc_motor_clamping(self):
        lstm_acts = [a for a in self.result["actuator_info"] if a["controller_type"] == "ControllerNeuralLSTM"]
        for a in lstm_acts:
            self.assertIn("ClampingDCMotor", a["clamping_types"])


def test_sync_torque_telemetry_reads_backend_effort_buffers_in_user_order() -> None:
    """Report torque telemetry in public joint order from backend-order effort buffers."""
    joint_pos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    joint_vel = wp.zeros_like(joint_pos)
    joint_pos_target = wp.zeros_like(joint_pos)
    joint_vel_target = wp.zeros_like(joint_pos)
    joint_stiffness = wp.zeros_like(joint_pos)
    joint_damping = wp.zeros_like(joint_pos)
    effort_limit = wp.full((1, 3), 1000.0, dtype=wp.float32, device="cpu")
    joint_modes = wp.array(np.asarray([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    sim_bind_joint_effort = wp.array(
        np.asarray([[100.0, 200.0, 300.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    actuator_computed_effort = wp.array(
        np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    computed = wp.zeros_like(joint_pos)
    applied = wp.zeros_like(joint_pos)

    wp.launch(
        sync_torque_telemetry,
        dim=joint_pos.shape,
        inputs=[
            joint_pos,
            joint_vel,
            joint_pos_target,
            joint_vel_target,
            joint_stiffness,
            joint_damping,
            effort_limit,
            joint_modes,
            sim_bind_joint_effort,
            actuator_computed_effort,
            user_to_backend,
            True,
        ],
        outputs=[computed, applied],
        device="cpu",
    )

    np.testing.assert_allclose(computed.numpy(), np.asarray([[30.0, 100.0, 20.0]], dtype=np.float32))
    np.testing.assert_allclose(applied.numpy(), np.asarray([[300.0, 100.0, 200.0]], dtype=np.float32))


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


if __name__ == "__main__":
    unittest.main()
