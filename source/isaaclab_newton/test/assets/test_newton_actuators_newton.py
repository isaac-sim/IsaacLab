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

import json
import os
import tempfile
import unittest

import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationCfg, build_simulation_context

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from isaaclab_assets import ANYMAL_C_CFG

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
# Actuator configurations under test
# ---------------------------------------------------------------------------

IDEAL_PD_ACTUATORS = {
    "legs": IdealPDActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
        effort_limit=80.0,
    ),
}

DC_MOTOR_ACTUATORS = {
    "legs": DCMotorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        saturation_effort=120.0,
        effort_limit=80.0,
        velocity_limit=7.5,
        stiffness={".*": 40.0},
        damping={".*": 5.0},
    ),
}

MIXED_ACTUATORS = {
    "hips": IdealPDActuatorCfg(
        joint_names_expr=[".*HAA"],
        stiffness=40.0,
        damping=5.0,
        effort_limit=80.0,
    ),
    "knees": DCMotorCfg(
        joint_names_expr=[".*HFE", ".*KFE"],
        saturation_effort=120.0,
        effort_limit=80.0,
        velocity_limit=7.5,
        stiffness={".*": 40.0},
        damping={".*": 5.0},
    ),
}

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
) -> dict:
    """Run ANYmal-C and return recorded trajectories + telemetry.

    Always records ``joint_pos``, ``joint_vel``, ``computed_torque``, and
    ``applied_torque`` so callers don't need a separate "with telemetry"
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

    Returns:
        Dict with ``joint_pos``, ``joint_vel``, ``computed_torque``,
        ``applied_torque`` (lists of per-step tensors) plus ``target_pos``
        and ``target_vel`` snapshots.
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
        art_cfg = ANYMAL_C_CFG.replace(actuators=actuators, prim_path="/World/Env_.*/Robot")
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

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)
        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)
        if feedforward is not None:
            articulation.set_joint_effort_target_index(
                target=torch.full_like(init_pos, feedforward),
            )

        recorded_pos, recorded_vel = [], []
        recorded_computed, recorded_applied = [], []
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
            recorded_computed.append(wp.to_torch(articulation.data.computed_torque).clone())
            recorded_applied.append(wp.to_torch(articulation.data.applied_torque).clone())

    return {
        "joint_pos": recorded_pos,
        "joint_vel": recorded_vel,
        "computed_torque": recorded_computed,
        "applied_torque": recorded_applied,
        "target_pos": target_pos.clone(),
        "target_vel": target_vel.clone(),
    }


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class _EquivalenceTestBase(unittest.TestCase):
    """Base for Lab-vs-Newton equivalence tests.

    Subclasses set ``actuators`` to the config under test.  ``setUpClass``
    runs the simulation with both ``use_newton_actuators=False`` (Lab path)
    and ``True`` (Newton path) and stores the results.
    """

    __test__ = False
    actuators: dict = {}
    feedforward: float | None = None
    dt: float = DT
    newton_cfg: NewtonCfg = NEWTON_CFG
    num_steps: int = NUM_STEPS
    decimation: int = 1
    pos_atol: float = 2e-3
    pos_rtol: float = 1e-3
    vel_atol: float = 1e-2
    vel_rtol: float = 1e-2
    torque_atol: float = 1e-3
    torque_rtol: float = 1e-3

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

    def test_joint_positions_match(self):
        for step_i, (lab, newton) in enumerate(zip(self.lab_result["joint_pos"], self.newton_result["joint_pos"])):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.pos_atol,
                rtol=self.pos_rtol,
                msg=f"Joint positions diverged at step {step_i}",
            )

    def test_joint_velocities_match(self):
        for step_i, (lab, newton) in enumerate(zip(self.lab_result["joint_vel"], self.newton_result["joint_vel"])):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.vel_atol,
                rtol=self.vel_rtol,
                msg=f"Joint velocities diverged at step {step_i}",
            )

    def test_applied_torque_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["applied_torque"], self.newton_result["applied_torque"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.torque_atol,
                rtol=self.torque_rtol,
                msg=f"applied_torque diverged at step {step_i}",
            )

    def test_computed_torque_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["computed_torque"], self.newton_result["computed_torque"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.torque_atol,
                rtol=self.torque_rtol,
                msg=f"computed_torque diverged at step {step_i}",
            )


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


class TestMixedActuatorEquivalence(_EquivalenceTestBase):
    """Mixed actuators (IdealPD on HAA, DCMotor on HFE/KFE): Lab vs Newton."""

    __test__ = True
    actuators = MIXED_ACTUATORS


MIXED_WITH_IMPLICIT_ACTUATORS = {
    "hips": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA"],
        stiffness=40.0,
        damping=5.0,
    ),
    "thighs": IdealPDActuatorCfg(
        joint_names_expr=[".*HFE"],
        stiffness=40.0,
        damping=5.0,
        effort_limit=80.0,
    ),
    "knees": DCMotorCfg(
        joint_names_expr=[".*KFE"],
        saturation_effort=120.0,
        effort_limit=80.0,
        velocity_limit=7.5,
        stiffness=40.0,
        damping=5.0,
    ),
}


class TestMixedWithImplicitEquivalence(_EquivalenceTestBase):
    """Implicit HAA + IdealPD HFE + DCMotor KFE: Lab vs Newton.

    Verifies that implicit actuators (handled by the physics engine's
    built-in joint drives) coexist correctly with explicit Newton actuators.
    """

    __test__ = True
    actuators = MIXED_WITH_IMPLICIT_ACTUATORS


# ---------------------------------------------------------------------------
# Implicit-only fast-path: enable Newton actuator branch with no explicit groups
# ---------------------------------------------------------------------------

IMPLICIT_ONLY_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
    ),
}


class TestImplicitOnlyEquivalence(_EquivalenceTestBase):
    """All-implicit articulation with ``use_newton_actuators=True``: Lab vs fast-path."""

    __test__ = True
    actuators = IMPLICIT_ONLY_ACTUATORS


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


CARTPOLE_EXPLICIT_ACTUATORS = {
    "all_joints": IdealPDActuatorCfg(
        joint_names_expr=["slider_to_cart", "cart_to_pole"],
        stiffness=10.0,
        damping=1.0,
        effort_limit=100.0,
    ),
}


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

        anymal_cfg = ANYMAL_C_CFG.replace(actuators=IDEAL_PD_ACTUATORS, prim_path="/World/Env_.*/Anymal")
        cartpole_cfg = CARTPOLE_CFG.replace(
            actuators=CARTPOLE_EXPLICIT_ACTUATORS,
            prim_path="/World/Env_.*/Cartpole",
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


class _MockScene:
    """Minimal stand-in for ``InteractiveScene`` accepted by ``ManagerTermBase``."""

    def __init__(self, assets: dict, num_envs: int):
        self._assets = assets
        self.num_envs = num_envs

    def __getitem__(self, name: str):
        return self._assets[name]


class _MockEnv:
    """Minimal stand-in for ``ManagerBasedEnv`` for invoking DR terms.

    ``randomize_actuator_gains`` only reads ``env.scene[name]`` and
    ``env.scene.num_envs`` (plus ``env.num_envs`` / ``env.device`` from the
    ``ManagerTermBase`` properties). No simulator access is needed because
    the DR term reaches the actuator adapter via ``self.asset.newton_actuator_adapter``.
    """

    def __init__(self, assets: dict, num_envs: int, device: str):
        self.scene = _MockScene(assets, num_envs)
        self.num_envs = num_envs
        self.device = device


def _build_dr_term(env, asset_name, joint_ids=None):
    from isaaclab.envs.mdp.events import randomize_actuator_gains  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: PLC0415

    asset_cfg = SceneEntityCfg(asset_name)
    if joint_ids is not None:
        asset_cfg.joint_ids = joint_ids
    cfg = EventTermCfg(
        func=randomize_actuator_gains,
        params={
            "asset_cfg": asset_cfg,
            "stiffness_distribution_params": (100.0, 100.0),
            "damping_distribution_params": (5.0, 5.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    return randomize_actuator_gains(cfg, env), asset_cfg


class TestRandomizeActuatorGainsViaEventsNewton(unittest.TestCase):
    """End-to-end DR test for the Newton backend.

    Drives ``randomize_actuator_gains`` (events.py) and verifies the new
    kp/kd values land on the controllers of the articulation's Newton
    actuators — exercising the full path: events →
    ``write_actuator_stiffness_to_sim`` → per-actuator
    ``ArticulationView.set_actuator_parameter`` (with the per-DOF mapping
    silently skipping actuators that belong to other articulations).

    With ``operation="abs"`` and ``distribution="uniform"`` over a
    degenerate range ``(K, K)``, every randomized cell is set to exactly
    ``K`` — so the assertions are deterministic.
    """

    @staticmethod
    def _gather_param(articulation, attr) -> torch.Tensor:
        """Read ``controller.<attr>`` for every Newton actuator via the view.

        Iterates the global adapter's actuator list. ``get_actuator_parameter``
        returns zeros for DOFs that don't belong to this articulation's
        view (the per-DOF mapping skips them), so summing across all
        actuators yields a clean ``(num_envs, num_joints)`` snapshot for
        this articulation.
        """
        n_env = articulation.num_instances
        n_j = articulation.num_joints
        out = torch.zeros((n_env, n_j), device=articulation.device)
        adapter = SimulationManager._adapter
        if adapter is None:
            return out
        for act in adapter.actuators:
            ctrl = act.controller
            if not hasattr(ctrl, attr):
                continue
            cur_wp = articulation._root_view.get_actuator_parameter(act, ctrl, attr)
            out += wp.to_torch(cur_wp)
        return out

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
                prim_path="/World/Env_.*/Robot",
            )
            anymal = Articulation(art_cfg)
            sim.reset()

            adapter = SimulationManager._adapter
            self.assertIsNotNone(adapter, "Newton adapter should exist with use_newton_actuators=True")
            kp_before = self._gather_param(anymal, "kp").clone()
            kd_before = self._gather_param(anymal, "kd").clone()

            env = _MockEnv({"robot": anymal}, NUM_ENVS, anymal.device)
            term, asset_cfg = _build_dr_term(env, "robot")
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

            kp_after = self._gather_param(anymal, "kp")
            kd_after = self._gather_param(anymal, "kd")
            n = anymal.num_joints
            torch.testing.assert_close(kp_after[0], torch.full((n,), 100.0, device=anymal.device))
            torch.testing.assert_close(kd_after[0], torch.full((n,), 5.0, device=anymal.device))
            # Other envs untouched.
            for env_idx in range(1, NUM_ENVS):
                torch.testing.assert_close(kp_after[env_idx], kp_before[env_idx])
                torch.testing.assert_close(kd_after[env_idx], kd_before[env_idx])

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

            anymal_cfg = ANYMAL_C_CFG.replace(actuators=IDEAL_PD_ACTUATORS, prim_path="/World/Env_.*/Anymal")
            cartpole_cfg = CARTPOLE_CFG.replace(
                actuators=CARTPOLE_EXPLICIT_ACTUATORS,
                prim_path="/World/Env_.*/Cartpole",
            )
            cartpole_cfg.init_state = cartpole_cfg.init_state.replace(pos=(0.0, 3.0, 2.0))
            anymal = Articulation(anymal_cfg)
            cartpole = Articulation(cartpole_cfg)
            sim.reset()

            self.assertIsNotNone(SimulationManager._adapter)

            anymal_kp_before = self._gather_param(anymal, "kp").clone()
            anymal_kd_before = self._gather_param(anymal, "kd").clone()
            cp_kp_before = self._gather_param(cartpole, "kp").clone()
            cp_kd_before = self._gather_param(cartpole, "kd").clone()

            env = _MockEnv({"anymal": anymal, "cartpole": cartpole}, NUM_ENVS, anymal.device)
            term, asset_cfg = _build_dr_term(env, "cartpole")
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

            cp_kp_after = self._gather_param(cartpole, "kp")
            cp_kd_after = self._gather_param(cartpole, "kd")
            n_cp = cartpole.num_joints
            torch.testing.assert_close(cp_kp_after[0], torch.full((n_cp,), 100.0, device=anymal.device))
            torch.testing.assert_close(cp_kd_after[0], torch.full((n_cp,), 5.0, device=anymal.device))

            # ANYmal is untouched (DR was scoped to cartpole).
            torch.testing.assert_close(self._gather_param(anymal, "kp"), anymal_kp_before)
            torch.testing.assert_close(self._gather_param(anymal, "kd"), anymal_kd_before)

            # Cartpole's other envs are also untouched (env_ids=[0] only).
            for env_idx in range(1, NUM_ENVS):
                torch.testing.assert_close(cp_kp_after[env_idx], cp_kp_before[env_idx])
                torch.testing.assert_close(cp_kd_after[env_idx], cp_kd_before[env_idx])


# ---------------------------------------------------------------------------
# DelayedPD equivalence: PD with actuator command delay
# ---------------------------------------------------------------------------

DELAYED_PD_ACTUATORS = {
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
        effort_limit=80.0,
        min_delay=2,
        max_delay=4,
    ),
}


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


class TestDecimationIdealPD(_DecimationMixin, TestIdealPDEquivalence):
    """IdealPD — decimation=2 + CUDA graph."""


class TestDecimationDelayedPD(_DecimationMixin, TestDelayedPDEquivalence):
    """DelayedPD — decimation=2 + CUDA graph (delay queue stepped inside the captured graph)."""


class TestDecimationMixed(_DecimationMixin, TestMixedActuatorEquivalence):
    """Mixed (IdealPD + DCMotor) — decimation=2 + CUDA graph."""


# ---------------------------------------------------------------------------
# Per-env reset: actuator state isolation
# ---------------------------------------------------------------------------

RESET_WARMUP_STEPS = 3


class TestActuatorStateReset(unittest.TestCase):
    """Reset must clear the actuator state buffers for the requested envs only.

    Inspects ``adapter.actuators[i].state.delay_state.num_pushes`` directly:

    * After warmup, ``num_pushes > 0`` for every DOF (buffer was populated).
    * After ``articulation.reset(env_ids=[0])``, the entries for env 0's DOFs
      must be ``0`` and the entries for env 1's DOFs must remain ``> 0``.

    Done independently on Lab and Newton paths. Lab inspects the
    ``positions_delay_buffer._circular_buffer`` of its DelayedPDActuator;
    Newton inspects the model-wide adapter's per-actuator state.
    """

    RESET_ENV: int = 0
    UNCHANGED_ENV: int = 1

    def _build_and_warm(self, *, use_newton_actuators: bool):
        sim_cfg = SimulationCfg(
            dt=DT,
            physics=NEWTON_CFG,
            use_newton_actuators=use_newton_actuators,
        )
        ctx = build_simulation_context(
            device="cuda:0",
            gravity_enabled=True,
            add_ground_plane=True,
            sim_cfg=sim_cfg,
        )
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        for i in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
        art_cfg = ANYMAL_C_CFG.replace(
            actuators=DELAYED_PD_ACTUATORS,
            prim_path="/World/Env_.*/Robot",
        )
        articulation = Articulation(art_cfg)
        sim.reset()

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)
        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)
        for _ in range(RESET_WARMUP_STEPS):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)
        return ctx, sim, articulation

    def test_newton_state_reset_isolated_to_reset_env(self):
        """Newton: ``num_pushes`` zeroes for env 0's DOFs only after reset of [0]."""
        ctx, sim, articulation = self._build_and_warm(use_newton_actuators=True)
        try:
            adapter = SimulationManager._adapter
            self.assertIsNotNone(adapter)
            # Find a DelayedPD actuator (it's the only one with delay_state).
            stateful_pairs = [
                (act, st)
                for act, st in zip(adapter.actuators, adapter._states_a)
                if st is not None and getattr(st, "delay_state", None) is not None
            ]
            self.assertGreater(len(stateful_pairs), 0, "expected at least one DelayedPD actuator with delay_state")

            # Per-DOF entry layout inside each actuator's state: ``act.indices``
            # is the flat global DOF id; envs are stacked so env 0's DOFs come first.
            for act, state in stateful_pairs:
                pushes_before = state.delay_state.num_pushes.numpy()
                self.assertTrue(
                    (pushes_before > 0).all(),
                    "expected non-zero num_pushes for all DOFs after warmup",
                )

            articulation.reset(env_ids=torch.tensor([self.RESET_ENV], device=articulation.device, dtype=torch.long))

            # Map each entry of ``act.indices`` to its env via the adapter's full
            # per-env DOF count (model.joint_dof_count // num_envs — includes free
            # joint DOFs on floating-base articulations, unlike articulation.num_joints
            # which counts only actuated DOFs).
            for act, state in stateful_pairs:
                pushes_after = state.delay_state.num_pushes.numpy()
                indices_np = act.indices.numpy()
                for i, global_dof in enumerate(indices_np):
                    env = int(global_dof) // adapter.num_joints
                    if env == self.RESET_ENV:
                        self.assertEqual(
                            int(pushes_after[i]),
                            0,
                            f"DOF {i} (env {env}) should be reset to 0, got {pushes_after[i]}",
                        )
                    else:
                        self.assertGreater(
                            int(pushes_after[i]),
                            0,
                            f"DOF {i} (env {env}) was NOT in reset env_ids but num_pushes is 0",
                        )
        finally:
            ctx.__exit__(None, None, None)

    def test_lab_state_reset_isolated_to_reset_env(self):
        """Lab: DelayedPDActuator circular buffer zeroed for env 0 only."""
        ctx, sim, articulation = self._build_and_warm(use_newton_actuators=False)
        try:
            from isaaclab.actuators import DelayedPDActuator  # noqa: PLC0415

            delayed = [a for a in articulation.actuators.values() if isinstance(a, DelayedPDActuator)]
            self.assertGreater(len(delayed), 0, "expected at least one Lab DelayedPDActuator")
            actuator = delayed[0]
            buf = actuator.positions_delay_buffer._circular_buffer._buffer
            # ``_buffer`` shape: (max_length, batch_size, num_joints).
            self.assertIsNotNone(buf, "delay buffer should be populated after warmup")
            self.assertTrue(
                (buf[:, self.UNCHANGED_ENV] != 0).any().item(),
                "expected non-zero buffer entries for env 1 after warmup",
            )

            articulation.reset(env_ids=torch.tensor([self.RESET_ENV], device=articulation.device, dtype=torch.long))

            self.assertTrue(
                torch.all(buf[:, self.RESET_ENV] == 0).item(),
                f"Lab: env {self.RESET_ENV} buffer not zeroed after reset.",
            )
            self.assertTrue(
                (buf[:, self.UNCHANGED_ENV] != 0).any().item(),
                f"Lab: env {self.UNCHANGED_ENV} buffer was zeroed — reset leaked into an unselected env.",
            )
        finally:
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# RemotizedPD actuator: PD + delay + position-based clamping lookup table
# ---------------------------------------------------------------------------

SPOT_KNEE_LOOKUP = [
    [-2.792900, -24.776718, 37.165077],
    [-2.767442, -26.290108, 39.435162],
    [-2.741984, -27.793369, 41.690054],
    [-2.716526, -29.285997, 43.928996],
    [-2.691068, -30.767536, 46.151304],
    [-2.665610, -32.237423, 48.356134],
    [-2.640152, -33.695168, 50.542751],
    [-2.614694, -35.140221, 52.710331],
    [-2.589236, -36.572052, 54.858078],
    [-2.563778, -37.990086, 56.985128],
    [-2.538320, -39.393730, 59.090595],
    [-2.512862, -40.782406, 61.173609],
    [-2.487404, -42.155487, 63.233231],
    [-2.461946, -43.512371, 65.268557],
    [-2.436488, -44.852371, 67.278557],
    [-2.411030, -46.174873, 69.262310],
    [-2.385572, -47.479156, 71.218735],
    [-2.360114, -48.764549, 73.146824],
    [-2.334656, -50.030334, 75.045502],
    [-2.309198, -51.275761, 76.913641],
    [-2.283740, -52.500103, 78.750154],
    [-2.258282, -53.702587, 80.553881],
    [-2.232824, -54.882442, 82.323664],
    [-2.207366, -56.038860, 84.058290],
    [-2.181908, -57.171028, 85.756542],
    [-2.156450, -58.278133, 87.417200],
    [-2.130992, -59.359314, 89.038971],
    [-2.105534, -60.413738, 90.620607],
    [-2.080076, -61.440529, 92.160793],
    [-2.054618, -62.438812, 93.658218],
    [-2.029160, -63.407692, 95.111538],
    [-2.003702, -64.346268, 96.519402],
    [-1.978244, -65.253670, 97.880505],
    [-1.952786, -66.128944, 99.193417],
    [-1.927328, -66.971176, 100.456764],
    [-1.901870, -67.779457, 101.669186],
    [-1.876412, -68.552864, 102.829296],
    [-1.850954, -69.290451, 103.935677],
    [-1.825496, -69.991325, 104.986988],
    [-1.800038, -70.654541, 105.981812],
    [-1.774580, -71.279190, 106.918785],
    [-1.749122, -71.864319, 107.796478],
    [-1.723664, -72.409088, 108.613632],
    [-1.698206, -72.912567, 109.368851],
    [-1.672748, -73.373871, 110.060806],
    [-1.647290, -73.792130, 110.688194],
    [-1.621832, -74.166512, 111.249767],
    [-1.596374, -74.496147, 111.744221],
    [-1.570916, -74.780251, 112.170376],
    [-1.545458, -75.017998, 112.526997],
    [-1.520000, -75.208656, 112.812984],
    [-1.494542, -75.351448, 113.027172],
    [-1.469084, -75.445686, 113.168530],
    [-1.443626, -75.490677, 113.236015],
    [-1.418168, -75.485771, 113.228657],
    [-1.392710, -75.430344, 113.145515],
    [-1.367252, -75.323830, 112.985744],
    [-1.341794, -75.165688, 112.748531],
    [-1.316336, -74.955406, 112.433109],
    [-1.290878, -74.692551, 112.038826],
    [-1.265420, -74.376694, 111.565041],
    [-1.239962, -74.007477, 111.011215],
    [-1.214504, -73.584579, 110.376869],
    [-1.189046, -73.107742, 109.661613],
    [-1.163588, -72.576752, 108.865128],
    [-1.138130, -71.991455, 107.987183],
    [-1.112672, -71.351707, 107.027561],
    [-1.087214, -70.657486, 105.986229],
    [-1.061756, -69.908813, 104.863220],
    [-1.036298, -69.105721, 103.658581],
    [-1.010840, -68.248337, 102.372505],
    [-0.985382, -67.336861, 101.005291],
    [-0.959924, -66.371513, 99.557270],
    [-0.934466, -65.352615, 98.028923],
    [-0.909008, -64.280533, 96.420799],
    [-0.883550, -63.155693, 94.733540],
    [-0.858092, -61.978588, 92.967882],
    [-0.832634, -60.749775, 91.124662],
    [-0.807176, -59.469845, 89.204767],
    [-0.781718, -58.139503, 87.209255],
    [-0.756260, -56.759487, 85.139231],
    [-0.730802, -55.330616, 82.995924],
    [-0.705344, -53.853729, 80.780594],
    [-0.679886, -52.329796, 78.494694],
    [-0.654428, -50.759762, 76.139643],
    [-0.628970, -49.144699, 73.717049],
    [-0.603512, -47.485737, 71.228605],
    [-0.578054, -45.784004, 68.676006],
    [-0.552596, -44.040764, 66.061146],
    [-0.527138, -42.257267, 63.385900],
    [-0.501680, -40.434883, 60.652325],
    [-0.476222, -38.574947, 57.862421],
    [-0.450764, -36.678982, 55.018473],
    [-0.425306, -34.748432, 52.122648],
    [-0.399848, -32.784836, 49.177254],
    [-0.374390, -30.789810, 46.184715],
    [-0.348932, -28.764952, 43.147428],
    [-0.323474, -26.711969, 40.067954],
    [-0.298016, -24.632576, 36.948864],
    [-0.272558, -22.528547, 33.792821],
    [-0.247100, -20.401667, 30.602500],
]
"""Spot knee joint parameter lookup table (102 entries).

Columns: joint angle [rad], transmission ratio, output torque [N*m].
Sourced from :mod:`isaaclab_assets.robots.spot`.
"""


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
            prim_path="/World/Env_.*/Robot",
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

    Uses the Spot knee lookup table (102 entries) on ANYmal's KFE joints,
    with IdealPD on HAA and HFE joints.
    """

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

        cls.result = _run_authoring_introspection(
            {
                "hips": IdealPDActuatorCfg(
                    joint_names_expr=[".*HAA", ".*HFE"],
                    stiffness=40.0,
                    damping=5.0,
                    effort_limit=80.0,
                ),
                "knees": RemotizedPDActuatorCfg(
                    joint_names_expr=[".*KFE"],
                    stiffness=60.0,
                    damping=1.5,
                    effort_limit=80.0,
                    max_delay=3,
                    joint_parameter_lookup=SPOT_KNEE_LOOKUP,
                ),
            }
        )

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
        from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

        cls.actuators = {
            "hips": IdealPDActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE"],
                stiffness=40.0,
                damping=5.0,
                effort_limit=80.0,
            ),
            "knees": RemotizedPDActuatorCfg(
                joint_names_expr=[".*KFE"],
                stiffness=60.0,
                damping=1.5,
                effort_limit=80.0,
                max_delay=3,
                joint_parameter_lookup=SPOT_KNEE_LOOKUP,
            ),
        }
        super().setUpClass()


class TestDecimationRemotizedPD(_DecimationMixin, TestRemotizedPDEquivalence):
    """RemotizedPD — decimation=2 + CUDA graph."""


class TestManagerBasedSceneNewtonActuatorAuthoring(unittest.TestCase):
    """Regression test for Newton actuator authoring in manager-based clone paths.

    The default G1 config uses ``ImplicitActuatorCfg`` for every group, which
    intentionally skips ``NewtonActuator`` USD authoring. To exercise the
    authoring path we override the scene's robot actuators with explicit
    ``DCMotorCfg`` groups covering the same joint patterns.
    """

    def test_newton_actuators_present_for_g1_manager_env(self):
        env_cfg = G1FlatEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.decimation = 1
        env_cfg.scene.contact_forces = None
        env_cfg.rewards.feet_air_time = None
        env_cfg.rewards.feet_slide = None
        env_cfg.terminations.base_contact = None
        env_cfg.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=95,
                nconmax=10,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        )
        env_cfg.sim.use_newton_actuators = True
        env_cfg.scene.robot.actuators = {
            "legs": DCMotorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "torso_joint",
                ],
                saturation_effort=300.0,
                effort_limit=300.0,
                velocity_limit=20.0,
                stiffness=150.0,
                damping=5.0,
            ),
            "feet": DCMotorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                saturation_effort=20.0,
                effort_limit=20.0,
                velocity_limit=20.0,
                stiffness=20.0,
                damping=2.0,
            ),
            "arms": DCMotorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
                saturation_effort=300.0,
                effort_limit=300.0,
                velocity_limit=20.0,
                stiffness=40.0,
                damping=10.0,
            ),
        }
        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            stage = env.unwrapped.sim.stage
            actuator_prim_count = sum(1 for prim in stage.Traverse() if prim.GetTypeName() == "NewtonActuator")
            self.assertGreater(
                actuator_prim_count,
                0,
                "Expected authored NewtonActuator prims in manager-based scene workflow.",
            )
            self.assertGreater(
                len(SimulationManager.get_model().actuators),
                0,
                "Expected Newton model actuators to be non-empty with use_newton_actuators=True.",
            )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Neural network actuator authoring: MLP and LSTM
# ---------------------------------------------------------------------------


def _make_dummy_mlp_checkpoint(device: str = "cpu") -> str:
    """Create a minimal TorchScript MLP checkpoint with metadata.

    The network accepts 6 inputs (3 history steps x 2 features per step
    in pos_vel order) and outputs 1 effort.
    """
    torch.manual_seed(42)
    net = (
        torch.nn.Sequential(
            torch.nn.Linear(6, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 1),
        )
        .to(device)
        .eval()
    )
    scripted = torch.jit.script(net)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    extra = {
        "metadata.json": json.dumps(
            {
                "model_type": "mlp",
                "input_order": "pos_vel",
                "input_idx": [0, 1, 2],
                "pos_scale": 1.0,
                "vel_scale": 0.5,
                "torque_scale": 2.0,
            }
        )
    }
    torch.jit.save(scripted, tmp_path, _extra_files=extra)
    return tmp_path


class _DummyLSTM(torch.nn.Module):
    """Minimal LSTM network for actuator testing."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(4, 1)

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hc_new = self.lstm(x, hc)
        return self.fc(out[:, -1, :]), hc_new


def _make_dummy_lstm_checkpoint(device: str = "cpu") -> str:
    """Create a minimal TorchScript LSTM checkpoint with metadata."""
    torch.manual_seed(42)
    net = _DummyLSTM().to(device).eval()
    scripted = torch.jit.script(net)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    extra = {"metadata.json": json.dumps({"model_type": "lstm"})}
    torch.jit.save(scripted, tmp_path, _extra_files=extra)
    return tmp_path


class TestNeuralMLPAuthoring(unittest.TestCase):
    """Verify ActuatorNetMLPCfg is authored as Newton NeuralMLP controller
    with DC motor clamping.
    """

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg  # noqa: PLC0415

        cls.mlp_path = _make_dummy_mlp_checkpoint()
        cls.result = _run_authoring_introspection(
            {
                "mlp_legs": ActuatorNetMLPCfg(
                    joint_names_expr=[".*HAA"],
                    network_file=cls.mlp_path,
                    saturation_effort=120.0,
                    effort_limit=80.0,
                    velocity_limit=7.5,
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
                    effort_limit=80.0,
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

        cls.lstm_path = _make_dummy_lstm_checkpoint()
        cls.result = _run_authoring_introspection(
            {
                "lstm_legs": ActuatorNetLSTMCfg(
                    joint_names_expr=[".*HAA"],
                    network_file=cls.lstm_path,
                    saturation_effort=120.0,
                    effort_limit=80.0,
                    velocity_limit=7.5,
                ),
                "pd_legs": IdealPDActuatorCfg(
                    joint_names_expr=[".*HFE", ".*KFE"],
                    stiffness=40.0,
                    damping=5.0,
                    effort_limit=80.0,
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


if __name__ == "__main__":
    unittest.main()
