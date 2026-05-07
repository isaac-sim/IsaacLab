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

import json
import os
import tempfile
import unittest

import torch
import warp as wp
from isaaclab_assets import ANYMAL_C_CFG
from isaaclab_physx.assets import Articulation
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ENVS = 2
NUM_STEPS = 10
DT = 1.0 / 120.0
TARGET_OFFSET = 0.1  # [rad] added to initial joint positions

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

# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _run_simulation(
    actuators: dict,
    use_newton_actuators: bool,
    *,
    num_steps: int = NUM_STEPS,
) -> dict:
    """Run ANYmal-C on PhysX and return recorded joint trajectories.

    Args:
        actuators: Actuator config dict overriding ANYmal's defaults.
        use_newton_actuators: Use Newton-native actuators (via
            :class:`PhysxActuatorWrapper`) when ``True``.
        num_steps: Number of simulation steps.

    Returns:
        Dict with ``joint_pos`` and ``joint_vel``, each a list of
        ``(NUM_ENVS, num_joints)`` tensors.
    """
    sim_cfg = SimulationCfg(
        dt=DT,
        physics=PhysxCfg(),
        use_newton_actuators=use_newton_actuators,
    )

    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None

        for i in range(NUM_ENVS):
            sim_utils.create_prim(
                f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0)
            )

        art_cfg = ANYMAL_C_CFG.replace(
            actuators=actuators,
            prim_path="/World/Env_.*/Robot",
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)

        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)

        recorded_pos, recorded_vel = [], []
        for _ in range(num_steps):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)

            recorded_pos.append(wp.to_torch(articulation.data.joint_pos).clone())
            recorded_vel.append(wp.to_torch(articulation.data.joint_vel).clone())

    return {"joint_pos": recorded_pos, "joint_vel": recorded_vel}


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class _EquivalenceTestBase(unittest.TestCase):
    """Base for Lab-vs-Newton equivalence tests on the PhysX backend.

    Subclasses set ``actuators`` to the config under test.  ``setUpClass``
    runs the simulation with both ``use_newton_actuators=False`` (Lab path)
    and ``True`` (Newton via PhysxActuatorWrapper) and stores the results.
    """

    __test__ = False
    actuators: dict = {}
    pos_atol: float = 2e-3
    pos_rtol: float = 1e-3
    vel_atol: float = 0.1
    vel_rtol: float = 1e-2

    @classmethod
    def setUpClass(cls):
        cls.lab_result = _run_simulation(cls.actuators, use_newton_actuators=False)
        cls.newton_result = _run_simulation(cls.actuators, use_newton_actuators=True)

    def test_joint_positions_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["joint_pos"], self.newton_result["joint_pos"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.pos_atol,
                rtol=self.pos_rtol,
                msg=f"Joint positions diverged at step {step_i}",
            )

    def test_joint_velocities_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["joint_vel"], self.newton_result["joint_vel"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.vel_atol,
                rtol=self.vel_rtol,
                msg=f"Joint velocities diverged at step {step_i}",
            )

    def test_trajectories_not_trivial(self):
        first = self.lab_result["joint_pos"][0]
        last = self.lab_result["joint_pos"][-1]
        diff = (last - first).abs().max().item()
        self.assertGreater(diff, 0.01, "Joints did not move — test is trivial")


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


class TestMixedActuatorEquivalence(_EquivalenceTestBase):
    """Mixed actuators (IdealPD on HAA, DCMotor on HFE/KFE): Lab vs Newton (PhysX)."""

    __test__ = True
    actuators = MIXED_ACTUATORS


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
# Implicit-only fast-path: enable Newton-actuator branch on PhysX with no explicit groups
# ---------------------------------------------------------------------------

IMPLICIT_ONLY_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
    ),
}


class TestImplicitOnlyEquivalencePhysx(_EquivalenceTestBase):
    """All-implicit articulation on PhysX with ``use_newton_actuators=True``: Lab vs fast-path."""

    __test__ = True
    actuators = IMPLICIT_ONLY_ACTUATORS


def _run_simulation_with_telemetry(
    actuators: dict,
    use_newton_actuators: bool,
    *,
    num_steps: int = NUM_STEPS,
) -> dict:
    """Like :func:`_run_simulation` but also records ``computed_torque`` / ``applied_torque``."""
    sim_cfg = SimulationCfg(
        dt=DT,
        physics=PhysxCfg(),
        use_newton_actuators=use_newton_actuators,
    )

    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None

        for i in range(NUM_ENVS):
            sim_utils.create_prim(
                f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0)
            )

        art_cfg = ANYMAL_C_CFG.replace(
            actuators=actuators,
            prim_path="/World/Env_.*/Robot",
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)

        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)

        recorded_pos, recorded_vel = [], []
        recorded_computed, recorded_applied = [], []
        for _ in range(num_steps):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)

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


class TestImplicitOnlyTelemetryPhysx(unittest.TestCase):
    """Implicit-only fast path on PhysX: shadow-PD telemetry matches the Lab formula."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation_with_telemetry(
            IMPLICIT_ONLY_ACTUATORS, use_newton_actuators=True,
        )
        cls.kp = 40.0
        cls.kd = 5.0

    def test_telemetry_is_nonzero(self):
        last = self.result["computed_torque"][-1]
        self.assertGreater(
            last.abs().max().item(),
            1e-3,
            "computed_torque is all-zero — implicit telemetry kernel did not run",
        )

    def test_telemetry_matches_pd_formula(self):
        target_q = self.result["target_pos"]
        target_v = self.result["target_vel"]
        for step_i, (q, qd, comp) in enumerate(
            zip(
                self.result["joint_pos"],
                self.result["joint_vel"],
                self.result["computed_torque"],
            )
        ):
            expected = self.kp * (target_q - q) + self.kd * (target_v - qd)
            torch.testing.assert_close(
                comp,
                expected,
                atol=5e-2,
                rtol=1e-2,
                msg=f"Telemetry diverged from PD formula at step {step_i}",
            )

    def test_applied_equals_computed_when_no_clip(self):
        for step_i, (comp, app) in enumerate(
            zip(self.result["computed_torque"], self.result["applied_torque"])
        ):
            torch.testing.assert_close(
                app,
                comp,
                atol=1e-5,
                rtol=1e-5,
                msg=f"applied_torque != computed_torque at step {step_i} (no clip expected)",
            )


class TestExplicitOnlyTelemetryPhysx(unittest.TestCase):
    """Explicit-only Newton actuators on PhysX: telemetry copies from the staging effort buffer."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation_with_telemetry(
            DC_MOTOR_ACTUATORS, use_newton_actuators=True,
        )

    def test_telemetry_is_nonzero(self):
        last = self.result["applied_torque"][-1]
        self.assertGreater(
            last.abs().max().item(),
            1e-3,
            "applied_torque is all-zero — explicit-DOF telemetry path did not run",
        )

    def test_computed_equals_applied_explicit(self):
        for step_i, (comp, app) in enumerate(
            zip(self.result["computed_torque"], self.result["applied_torque"])
        ):
            torch.testing.assert_close(
                comp,
                app,
                atol=1e-5,
                rtol=1e-5,
                msg=f"computed != applied for explicit-only at step {step_i}",
            )


class TestWriteActuatorGainsPhysx(unittest.TestCase):
    """``write_actuator_*_to_sim`` propagates kp/kd into Newton controllers (PhysX backend).

    Catches the silent no-op that ``randomize_actuator_gains`` would suffer
    from on PhysX with ``use_newton_actuators=True`` if these writers were
    missing (the events.py term falls back to ``hasattr`` and skips
    explicit actuators).
    """

    def test_writers_exist(self):
        # Guards against the missing-writer regression.
        from isaaclab_physx.assets import Articulation
        self.assertTrue(
            hasattr(Articulation, "write_actuator_stiffness_to_sim"),
            "Articulation is missing write_actuator_stiffness_to_sim — DR will silently no-op",
        )
        self.assertTrue(
            hasattr(Articulation, "write_actuator_damping_to_sim"),
            "Articulation is missing write_actuator_damping_to_sim — DR will silently no-op",
        )

    def test_writers_propagate_to_controller(self):
        sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=True)
        with build_simulation_context(
            device="cuda:0", gravity_enabled=True, add_ground_plane=True, sim_cfg=sim_cfg,
        ) as sim:
            sim._app_control_on_stop_handle = None
            for i in range(NUM_ENVS):
                sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
            art_cfg = ANYMAL_C_CFG.replace(
                actuators=DC_MOTOR_ACTUATORS, prim_path="/World/Env_.*/Robot",
            )
            articulation = Articulation(art_cfg)
            sim.reset()
            adapter = articulation.actuators["newton"]
            # Snapshot initial controller gains.
            kp_before = [
                wp.to_torch(a.controller.kp).clone() for a in adapter.actuators if hasattr(a.controller, "kp")
            ]
            kd_before = [
                wp.to_torch(a.controller.kd).clone() for a in adapter.actuators if hasattr(a.controller, "kd")
            ]
            self.assertGreater(len(kp_before), 0, "expected at least one PD controller in adapter")
            new_kp = adapter.stiffness.clone() * 2.0
            new_kd = adapter.damping.clone() * 3.0
            articulation.write_actuator_stiffness_to_sim(adapter, stiffness=new_kp)
            articulation.write_actuator_damping_to_sim(adapter, damping=new_kd)
            # Verify each controller's kp/kd actually changed (and roughly doubled/tripled).
            kp_idx = 0
            kd_idx = 0
            for newton_act in adapter.actuators:
                ctrl = newton_act.controller
                if hasattr(ctrl, "kp"):
                    after = wp.to_torch(ctrl.kp)
                    self.assertFalse(
                        torch.equal(after, kp_before[kp_idx]),
                        "controller.kp unchanged after write_actuator_stiffness_to_sim",
                    )
                    torch.testing.assert_close(after, kp_before[kp_idx] * 2.0, atol=1e-4, rtol=1e-4)
                    kp_idx += 1
                if hasattr(ctrl, "kd"):
                    after = wp.to_torch(ctrl.kd)
                    self.assertFalse(
                        torch.equal(after, kd_before[kd_idx]),
                        "controller.kd unchanged after write_actuator_damping_to_sim",
                    )
                    torch.testing.assert_close(after, kd_before[kd_idx] * 3.0, atol=1e-4, rtol=1e-4)
                    kd_idx += 1


# ---------------------------------------------------------------------------
# Partial environment reset: verify per-env reset equivalence
# ---------------------------------------------------------------------------

RESET_WARMUP_STEPS = 3
RESET_TOTAL_STEPS = 10


def _run_simulation_with_reset(
    actuators: dict,
    use_newton_actuators: bool,
) -> dict:
    """Run ANYmal-C on PhysX with a mid-simulation reset of env 0 only.

    Steps ``RESET_WARMUP_STEPS``, then resets env 0 to its initial joint state
    (zeroing velocity), then steps ``RESET_TOTAL_STEPS - RESET_WARMUP_STEPS``
    more. Returns per-step joint positions and velocities.

    This exercises the actuator state reset path (delay buffers, neural
    hidden states, etc.) for a subset of environments.

    Args:
        actuators: Actuator config dict overriding ANYmal's defaults.
        use_newton_actuators: Use Newton-native actuators when ``True``.

    Returns:
        Dict with ``joint_pos`` and ``joint_vel``, each a list of
        ``(NUM_ENVS, num_joints)`` tensors.
    """
    sim_cfg = SimulationCfg(
        dt=DT,
        physics=PhysxCfg(),
        use_newton_actuators=use_newton_actuators,
    )

    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None

        for i in range(NUM_ENVS):
            sim_utils.create_prim(
                f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0)
            )

        art_cfg = ANYMAL_C_CFG.replace(
            actuators=actuators,
            prim_path="/World/Env_.*/Robot",
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)

        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)

        recorded_pos, recorded_vel = [], []

        for step_i in range(RESET_TOTAL_STEPS):
            if step_i == RESET_WARMUP_STEPS:
                env_ids = torch.tensor([0], device="cuda:0")
                articulation.write_joint_position_to_sim_index(
                    position=init_pos[0:1], env_ids=env_ids,
                )
                articulation.write_joint_velocity_to_sim_index(
                    velocity=torch.zeros_like(init_pos[0:1]), env_ids=env_ids,
                )
                articulation.reset(env_ids=[0])

            articulation.write_data_to_sim()
            sim.step()
            articulation.update(DT)

            recorded_pos.append(wp.to_torch(articulation.data.joint_pos).clone())
            recorded_vel.append(wp.to_torch(articulation.data.joint_vel).clone())

    return {"joint_pos": recorded_pos, "joint_vel": recorded_vel}


class TestPartialResetEquivalence(unittest.TestCase):
    """Per-environment reset with DelayedPD actuators: Lab vs Newton (PhysX).

    Resets env 0 mid-simulation while env 1 continues uninterrupted.
    Uses DelayedPD actuators because they carry internal state (delay
    buffers) that must be properly reset per environment.

    Verifies:
    - Lab and Newton paths produce matching trajectories after partial reset.
    - The two environments diverge after the reset (proving it took effect).
    """

    @classmethod
    def setUpClass(cls):
        cls.lab_result = _run_simulation_with_reset(
            DELAYED_PD_ACTUATORS, use_newton_actuators=False,
        )
        cls.newton_result = _run_simulation_with_reset(
            DELAYED_PD_ACTUATORS, use_newton_actuators=True,
        )

    def test_joint_positions_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["joint_pos"], self.newton_result["joint_pos"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=2e-3,
                rtol=1e-3,
                msg=f"Positions diverged at step {step_i}",
            )

    def test_joint_velocities_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["joint_vel"], self.newton_result["joint_vel"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=0.1,
                rtol=1e-2,
                msg=f"Velocities diverged at step {step_i}",
            )

    def test_envs_diverge_after_reset(self):
        """After resetting env 0, the two envs must have different states."""
        post_reset_pos = self.lab_result["joint_pos"][RESET_WARMUP_STEPS + 1]
        diff = (post_reset_pos[0] - post_reset_pos[1]).abs().max().item()
        self.assertGreater(
            diff, 0.001,
            "Env 0 and env 1 are identical after partial reset — reset had no effect",
        )

    def test_trajectories_not_trivial(self):
        first = self.lab_result["joint_pos"][0]
        last = self.lab_result["joint_pos"][-1]
        diff = (last - first).abs().max().item()
        self.assertGreater(diff, 0.01, "Joints did not move — test is trivial")


# ---------------------------------------------------------------------------
# RemotizedPD authoring: PD + delay + position-based clamping lookup table
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


class TestRemotizedPDFunctional(unittest.TestCase):
    """Verify RemotizedPDActuatorCfg runs correctly on PhysX with Newton actuators.

    Uses the Spot knee lookup table (102 entries) on ANYmal's KFE joints.
    """

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

        cls.result = _run_simulation(
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
            },
            use_newton_actuators=True,
        )

    def test_trajectories_not_trivial(self):
        first = self.result["joint_pos"][0]
        last = self.result["joint_pos"][-1]
        diff = (last - first).abs().max().item()
        self.assertGreater(diff, 0.01, "Joints did not move — test is trivial")

    def test_positions_finite(self):
        for step_i, pos in enumerate(self.result["joint_pos"]):
            self.assertTrue(
                torch.isfinite(pos).all(),
                f"Non-finite positions at step {step_i}",
            )


# ---------------------------------------------------------------------------
# Neural network actuator authoring: MLP and LSTM
# ---------------------------------------------------------------------------


def _make_dummy_mlp_checkpoint(device: str = "cpu") -> str:
    """Create a minimal TorchScript MLP checkpoint with metadata."""
    torch.manual_seed(42)
    net = torch.nn.Sequential(
        torch.nn.Linear(6, 8),
        torch.nn.ELU(),
        torch.nn.Linear(8, 1),
    ).to(device).eval()
    scripted = torch.jit.script(net)

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    extra = {
        "metadata.json": json.dumps({
            "model_type": "mlp",
            "input_order": "pos_vel",
            "input_idx": [0, 1, 2],
            "pos_scale": 1.0,
            "vel_scale": 0.5,
            "torque_scale": 2.0,
        })
    }
    torch.jit.save(scripted, tmp.name, _extra_files=extra)
    return tmp.name


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

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    extra = {"metadata.json": json.dumps({"model_type": "lstm"})}
    torch.jit.save(scripted, tmp.name, _extra_files=extra)
    return tmp.name


class TestNeuralMLPFunctional(unittest.TestCase):
    """Verify ActuatorNetMLPCfg runs on PhysX with Newton actuators."""

    @classmethod
    def setUpClass(cls):
        from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg  # noqa: PLC0415

        cls.mlp_path = _make_dummy_mlp_checkpoint()
        cls.result = _run_simulation(
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
            },
            use_newton_actuators=True,
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.mlp_path)

    def test_trajectories_not_trivial(self):
        first = self.result["joint_pos"][0]
        last = self.result["joint_pos"][-1]
        diff = (last - first).abs().max().item()
        self.assertGreater(diff, 0.01, "Joints did not move — test is trivial")

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

        cls.lstm_path = _make_dummy_lstm_checkpoint()
        cls.result = _run_simulation(
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
            },
            use_newton_actuators=True,
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.lstm_path)

    def test_trajectories_not_trivial(self):
        first = self.result["joint_pos"][0]
        last = self.result["joint_pos"][-1]
        diff = (last - first).abs().max().item()
        self.assertGreater(diff, 0.01, "Joints did not move — test is trivial")

    def test_positions_finite(self):
        for step_i, pos in enumerate(self.result["joint_pos"]):
            self.assertTrue(
                torch.isfinite(pos).all(),
                f"Non-finite positions at step {step_i}",
            )


if __name__ == "__main__":
    unittest.main()
