# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for Newton-native actuators on the Newton physics backend.

The Newton backend always steps actuators through the physics engine:
explicit Lab actuator configs (IdealPD, DCMotor, DelayedPD, RemotizedPD,
ActuatorNet*) are authored as ``NewtonActuator`` USD prims and stepped by
Newton; implicit configs use the solver's built-in joint drives. These
tests assert that behavior directly:

* PD tracking: commanded poses are reached and held (gravity disabled).
* Torque telemetry: ``computed_torque`` follows the PD law and
  ``applied_torque`` follows each clamp's exact formula (max-effort box,
  DC-motor speed-torque curve, position-based lookup clamping).
* Delay semantics: a target switch takes effect exactly ``delay_steps``
  actuator steps later.
* Decimation folding: one folded manager step covering ``d`` sub-steps
  reproduces the trajectory of ``d`` explicit single sub-steps.
* Ordering, reset isolation, DR gain writes, and USD authoring.

Using ANYmal-C — a 12-DOF quadruped on a floating base — exercises the
coordinate-vs-DOF index separation that is critical when free joints shift
the mapping between ``joint_q`` (coordinate layout) and ``joint_qd``
(DOF layout).
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import json
import os
import re
import tempfile
import unittest

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.actuators.kernels import sync_torque_telemetry
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
# 200 Hz, the standard quadruped physics rate. At 120 Hz the explicit-Euler loop
# is marginally unstable for the ANYdrive gains on the knee joints (the lightest
# links): tracking runs drift 0.15-1.0 rad instead of settling.
DT = 1.0 / 200.0
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
    *,
    dt: float = DT,
    newton_cfg: NewtonCfg = NEWTON_CFG,
    num_steps: int = NUM_STEPS,
    decimation: int = 1,
    fold_decimation: bool = True,
    gravity_enabled: bool = True,
    add_ground_plane: bool = True,
    target_offset: float = TARGET_OFFSET,
    feedforward: float | None = None,
    retarget_offset: float | None = None,
    retarget_at_step: int | None = None,
    joint_ordering: tuple[str, ...] | None = None,
    permutation_sensitive_commands: bool = False,
) -> dict:
    """Run ANYmal-C on the Newton actuator path, record trajectories + telemetry.

    Always records ``joint_pos``, ``joint_vel``, ``computed_torque``, and
    ``applied_torque`` (public order) plus the adapter's backend-order effort
    buffers, the pre-run joint state, and the gain/limit snapshots that the
    telemetry kernel reads — everything the direct behavior assertions need.

    Args:
        actuators: Actuator config dict overriding ANYmal's defaults.
        dt: Physics timestep [s].
        newton_cfg: Newton physics configuration.
        num_steps: Number of policy-level steps.
        decimation: Actuator steps per policy step.
        fold_decimation: When ``True`` and ``decimation > 1``, ask the physics
            manager to fold the whole decimation loop into a single step call
            (CUDA-graph d-loop); otherwise an explicit Python inner loop
            drives ``decimation`` single sub-steps per policy step.
        gravity_enabled: Whether gravity is enabled.
        add_ground_plane: Whether a ground plane is spawned.
        target_offset: Offset added to the initial joint positions to form
            the position target [rad].
        feedforward: When not ``None``, set a constant per-DOF feedforward
            effort target.
        retarget_offset: Offset for the second-phase position target [rad].
        retarget_at_step: Policy step at which the position target switches
            to ``init + retarget_offset``. Used by the delay-semantics test.
        joint_ordering: Optional explicit public joint-name order.
        permutation_sensitive_commands: Whether to command distinct position, velocity, and effort values by
            physical joint name.

    Returns:
        Recorded joint-name metadata, commands, initial state, public
        trajectories and torque telemetry, backend-order adapter effort
        traces, and gain/limit snapshots.
    """
    # ``build_simulation_context`` only honors ``gravity_enabled`` when it builds
    # the config itself; with an explicit ``sim_cfg`` gravity must be set here.
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(dt=dt, physics=newton_cfg, gravity=gravity)
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=gravity_enabled,
        add_ground_plane=add_ground_plane,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None
        for i in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
        art_cfg = ANYMAL_C_CFG.replace(
            actuators=actuators,
            prim_path="/World/Env_.*/Robot",
            joint_ordering=joint_ordering,
        )
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        # Raw articulations have no env reset event: start from the asset's
        # default joint state instead of the USD-authored zero pose, whose
        # ``init + offset`` targets can fall outside reachable joint ranges.
        articulation.write_joint_position_to_sim_index(position=articulation.data.default_joint_pos.torch.clone())
        articulation.write_joint_velocity_to_sim_index(velocity=articulation.data.default_joint_vel.torch.clone())

        if fold_decimation and decimation > 1:
            SimulationManager.set_decimation(decimation)
        folds_decimation = fold_decimation and decimation > 1 and SimulationManager.handles_decimation()

        adapter_exists = SimulationManager._adapter is not None
        num_model_actuators = len(SimulationManager.get_model().actuators)

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
        init_vel = wp.to_torch(articulation.data.joint_vel).clone()
        joint_stiffness = wp.to_torch(articulation.data.joint_stiffness).clone()
        joint_damping = wp.to_torch(articulation.data.joint_damping).clone()
        joint_effort_limits = wp.to_torch(articulation.data.joint_effort_limits).clone()
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
            target_pos = init_pos + target_offset
            target_vel = torch.zeros_like(init_pos)
            effort_target = None if feedforward is None else torch.full_like(init_pos, feedforward)

        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)
        if effort_target is not None:
            articulation.set_joint_effort_target_index(target=effort_target)

        target_pos2 = None
        recorded_pos, recorded_vel = [], []
        recorded_computed, recorded_applied = [], []
        recorded_adapter_computed, recorded_adapter_applied = [], []
        for step_index in range(num_steps):
            if retarget_at_step is not None and step_index == retarget_at_step:
                target_pos2 = init_pos + retarget_offset
                articulation.set_joint_position_target_index(target=target_pos2)
            if folds_decimation:
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
            recorded_adapter_computed.append(wp.to_torch(articulation.data._sim_bind_joint_computed_effort).clone())
            recorded_adapter_applied.append(wp.to_torch(articulation.data._sim_bind_joint_effort).clone())

    return {
        "joint_names": joint_names,
        "backend_joint_names": backend_joint_names,
        "joint_ordering": joint_ordering_state,
        "adapter_joint_names": backend_joint_names,
        "adapter_exists": adapter_exists,
        "num_model_actuators": num_model_actuators,
        "folds_decimation": folds_decimation,
        "init_pos": init_pos.clone(),
        "init_vel": init_vel.clone(),
        "joint_stiffness": joint_stiffness,
        "joint_damping": joint_damping,
        "joint_effort_limits": joint_effort_limits,
        "joint_pos": recorded_pos,
        "joint_vel": recorded_vel,
        "computed_torque": recorded_computed,
        "applied_torque": recorded_applied,
        "adapter_computed_effort": recorded_adapter_computed,
        "adapter_applied_effort": recorded_adapter_applied,
        "target_pos": target_pos.clone(),
        "target_vel": target_vel.clone(),
        "target_pos2": None if target_pos2 is None else target_pos2.clone(),
        "retarget_at_step": retarget_at_step,
        "effort_target": None if effort_target is None else effort_target.clone(),
    }


_ORDERING_TRACE_FIELDS = (
    "joint_pos",
    "joint_vel",
    "computed_torque",
    "applied_torque",
    "adapter_computed_effort",
    "adapter_applied_effort",
)
_ORDERING_TRACE_TOLERANCES = {
    "joint_pos": (2e-3, 1e-3),
    "joint_vel": (1e-2, 1e-2),
    "computed_torque": (1e-3, 1e-3),
    "applied_torque": (1e-3, 1e-3),
    "adapter_computed_effort": (1e-3, 1e-3),
    "adapter_applied_effort": (1e-3, 1e-3),
}


def _canonicalize_ordering_result(result: dict, canonical_joint_names: tuple[str, ...]) -> dict:
    """Gather public and adapter traces into one physical joint-name order."""
    canonical_result = dict(result)
    for field_name in _ORDERING_TRACE_FIELDS:
        source_names = result["adapter_joint_names"] if field_name.startswith("adapter_") else result["joint_names"]
        source_indices = tuple(source_names.index(name) for name in canonical_joint_names)
        canonical_result[field_name] = [
            values.index_select(
                1,
                torch.tensor(source_indices, dtype=torch.long, device=values.device),
            )
            for values in result[field_name]
        ]

    public_indices = tuple(result["joint_names"].index(name) for name in canonical_joint_names)
    for field_name in ("target_pos", "target_vel", "effort_target"):
        values = result[field_name]
        if values is not None:
            canonical_result[field_name] = values.index_select(
                1,
                torch.tensor(public_indices, dtype=torch.long, device=values.device),
            )

    canonical_result["joint_names"] = canonical_joint_names
    canonical_result["backend_joint_names"] = canonical_joint_names
    canonical_result["adapter_joint_names"] = canonical_joint_names
    return canonical_result


def test_newton_actuator_rollout_matches_reversed_joint_ordering() -> None:
    """Match Newton-backend actuator traces under reversed public joint ordering."""
    identity_result = _run_simulation(
        IDEAL_PD_ACTUATORS,
        permutation_sensitive_commands=True,
    )
    requested_joint_names = tuple(reversed(identity_result["joint_names"]))
    reversed_result = _run_simulation(
        IDEAL_PD_ACTUATORS,
        joint_ordering=requested_joint_names,
        permutation_sensitive_commands=True,
    )

    assert identity_result["joint_names"] == identity_result["backend_joint_names"]
    assert reversed_result["joint_names"] == requested_joint_names
    assert reversed_result["backend_joint_names"] == identity_result["backend_joint_names"]

    installed_ordering = reversed_result["joint_ordering"]
    assert installed_ordering is not None
    assert not installed_ordering["is_identity"]
    assert installed_ordering["user_names"] == requested_joint_names
    assert installed_ordering["backend_names"] == identity_result["backend_joint_names"]
    expected_user_to_backend = tuple(
        identity_result["backend_joint_names"].index(name) for name in requested_joint_names
    )
    expected_backend_to_user = tuple(
        requested_joint_names.index(name) for name in identity_result["backend_joint_names"]
    )
    assert installed_ordering["user_to_backend_indices"] == expected_user_to_backend
    assert installed_ordering["backend_to_user_indices"] == expected_backend_to_user

    canonical_joint_names = tuple(identity_result["backend_joint_names"])
    identity_result = _canonicalize_ordering_result(identity_result, canonical_joint_names)
    reversed_result = _canonicalize_ordering_result(reversed_result, canonical_joint_names)

    assert identity_result["joint_names"] == reversed_result["joint_names"] == canonical_joint_names
    for field_name in ("target_pos", "target_vel", "effort_target"):
        torch.testing.assert_close(
            identity_result[field_name],
            reversed_result[field_name],
            rtol=0.0,
            atol=0.0,
            msg=f"{field_name} does not request the same physical command",
        )

    for field_name in _ORDERING_TRACE_FIELDS:
        atol, rtol = _ORDERING_TRACE_TOLERANCES[field_name]
        for step_index, (identity_values, reversed_values) in enumerate(
            zip(identity_result[field_name], reversed_result[field_name], strict=True)
        ):
            torch.testing.assert_close(
                identity_values,
                reversed_values,
                atol=atol,
                rtol=rtol,
                msg=f"{field_name} diverged at step {step_index}",
            )


# ---------------------------------------------------------------------------
# Telemetry formula helpers
#
# ``computed_torque`` / ``applied_torque`` are filled by the in-graph
# ``sync_torque_telemetry`` kernel right after the actuator step, i.e. from
# the joint state *before* the solver substeps of that step. With
# ``decimation == 1`` that state is exactly the state recorded after the
# previous policy step (or the initial state for step 0), so the recorded
# telemetry can be checked against the exact actuator formulas.
# ---------------------------------------------------------------------------


def _pre_step_states(result: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-step joint state each actuator step read: the state recorded after the previous step."""
    pos_seq = [result["init_pos"], *result["joint_pos"][:-1]]
    vel_seq = [result["init_vel"], *result["joint_vel"][:-1]]
    return pos_seq, vel_seq


def _matching_joint_ids(result: dict, patterns: list[str] | str) -> list[int]:
    """Public-order joint indices whose names fully match any of the given regexes."""
    if isinstance(patterns, str):
        patterns = [patterns]
    ids = [
        index
        for index, name in enumerate(result["joint_names"])
        if any(re.fullmatch(pattern, name) for pattern in patterns)
    ]
    assert ids, f"no joints matched {patterns}"
    return ids


def _assert_pd_computed_torque(
    result: dict,
    joint_ids: list[int],
    *,
    kp: float,
    kd: float,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> None:
    """``computed_torque`` must equal the PD law from the pre-step state on every step."""
    pos_prev, vel_prev = _pre_step_states(result)
    for step_index, computed in enumerate(result["computed_torque"]):
        expected = kp * (result["target_pos"] - pos_prev[step_index]) + kd * (
            result["target_vel"] - vel_prev[step_index]
        )
        torch.testing.assert_close(
            computed[:, joint_ids],
            expected[:, joint_ids],
            atol=atol,
            rtol=rtol,
            msg=f"computed_torque broke the PD law at step {step_index}",
        )


def _assert_max_effort_clamp(
    result: dict,
    joint_ids: list[int],
    *,
    effort_limit: float,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> None:
    """``applied_torque`` must equal ``computed_torque`` clamped to the symmetric effort box."""
    for step_index, (computed, applied) in enumerate(zip(result["computed_torque"], result["applied_torque"])):
        expected = torch.clamp(computed[:, joint_ids], min=-effort_limit, max=effort_limit)
        torch.testing.assert_close(
            applied[:, joint_ids],
            expected,
            atol=atol,
            rtol=rtol,
            msg=f"applied_torque broke the max-effort clamp at step {step_index}",
        )


def _dc_motor_effort_bounds(
    vel: torch.Tensor,
    *,
    saturation_effort: float,
    velocity_limit: float,
    effort_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newton ``ClampingDCMotor`` four-quadrant effort bounds for the given velocity."""
    corner_velocity = velocity_limit * (1.0 + effort_limit / saturation_effort)
    vel = torch.clamp(vel, min=-corner_velocity, max=corner_velocity)
    effort_max = torch.clamp(saturation_effort * (1.0 - vel / velocity_limit), max=effort_limit)
    effort_min = torch.clamp(saturation_effort * (-1.0 - vel / velocity_limit), min=-effort_limit)
    return effort_min, effort_max


def _assert_dc_motor_clamp(
    result: dict,
    joint_ids: list[int],
    *,
    saturation_effort: float,
    velocity_limit: float,
    effort_limit: float,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> None:
    """``applied_torque`` must equal ``computed_torque`` clamped by the DC-motor speed-torque curve."""
    _, vel_prev = _pre_step_states(result)
    for step_index, (computed, applied) in enumerate(zip(result["computed_torque"], result["applied_torque"])):
        effort_min, effort_max = _dc_motor_effort_bounds(
            vel_prev[step_index][:, joint_ids],
            saturation_effort=saturation_effort,
            velocity_limit=velocity_limit,
            effort_limit=effort_limit,
        )
        expected = torch.clamp(computed[:, joint_ids], min=effort_min, max=effort_max)
        torch.testing.assert_close(
            applied[:, joint_ids],
            expected,
            atol=atol,
            rtol=rtol,
            msg=f"applied_torque broke the DC-motor clamp at step {step_index}",
        )


def _assert_implicit_drive_telemetry(
    result: dict,
    joint_ids: list[int],
    *,
    atol: float = 1e-3,
    rtol: float = 1e-4,
) -> None:
    """Implicit DOFs report ``clamp(PD, ±effort_limit) + feedforward`` with computed == applied."""
    pos_prev, vel_prev = _pre_step_states(result)
    kp = result["joint_stiffness"][:, joint_ids]
    kd = result["joint_damping"][:, joint_ids]
    limit = result["joint_effort_limits"][:, joint_ids]
    ff = 0.0 if result["effort_target"] is None else result["effort_target"][:, joint_ids]
    for step_index, (computed, applied) in enumerate(zip(result["computed_torque"], result["applied_torque"])):
        pd = kp * (result["target_pos"][:, joint_ids] - pos_prev[step_index][:, joint_ids]) + kd * (
            result["target_vel"][:, joint_ids] - vel_prev[step_index][:, joint_ids]
        )
        expected = torch.clamp(pd, min=-limit, max=limit) + ff
        torch.testing.assert_close(
            computed[:, joint_ids],
            expected,
            atol=atol,
            rtol=rtol,
            msg=f"implicit computed_torque broke the drive law at step {step_index}",
        )
        torch.testing.assert_close(
            applied[:, joint_ids],
            computed[:, joint_ids],
            atol=0.0,
            rtol=0.0,
            msg=f"implicit applied_torque must mirror computed_torque at step {step_index}",
        )


def _assert_target_reached_and_held(
    test: unittest.TestCase,
    result: dict,
    *,
    hold_steps: int,
    tol: float,
    joint_ids: list[int] | None = None,
) -> None:
    """The commanded pose is reached within *tol* [rad] and held for the last *hold_steps* steps."""
    columns = slice(None) if joint_ids is None else joint_ids
    first_hold_step = len(result["joint_pos"]) - hold_steps
    for step_index, pos in enumerate(result["joint_pos"][-hold_steps:], start=first_hold_step):
        err = (pos[:, columns] - result["target_pos"][:, columns]).abs().max().item()
        test.assertLess(err, tol, f"pose error {err:.4f} rad exceeds {tol} rad at step {step_index}")


# Upstream defect (Newton MJWarp): the solver's implicit joint drives are baked
# into the MuJoCo actuator set at model finalize from the USD-authored drive
# gains. Assets authored with zero drive gains (e.g. ANYmal) therefore get no
# MuJoCo actuator at all, and the gains an :class:`ImplicitActuatorCfg` writes
# at initialization land in ``model.joint_target_ke`` where the solver never
# reads them back: the joints receive no drive torque. Reproduced identically
# on ``develop`` (this PR does not change implicit-drive handling). Remove the
# decorators when the solver honors post-finalize drive-gain updates.
_implicit_drive_inert_upstream = unittest.expectedFailure


# ---------------------------------------------------------------------------
# Direct behavior tests per actuator type
#
# Tracking runs disable gravity and skip the ground plane so the joint-space
# actuation is the only torque source and the commanded pose is the exact
# equilibrium; telemetry-law runs are self-consistent in any environment.
# ---------------------------------------------------------------------------

TRACKING_STEPS = 150
TRACKING_HOLD_STEPS = 30
TRACKING_OFFSET = 0.3  # [rad]
TRACKING_TOL = 0.05  # [rad]
DC_SATURATION_OFFSET = 2.5  # [rad]: initial PD demand kp * offset = 100 N*m > effort_limit


class TestIdealPDBehavior(unittest.TestCase):
    """IdealPDActuator on all 12 joints: PD tracking + exact telemetry laws."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            IDEAL_PD_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=TRACKING_OFFSET,
            num_steps=TRACKING_STEPS,
        )
        cls.joint_ids = _matching_joint_ids(cls.result, [".*HAA", ".*HFE", ".*KFE"])

    def test_commanded_pose_reached_and_held(self):
        _assert_target_reached_and_held(self, self.result, hold_steps=TRACKING_HOLD_STEPS, tol=TRACKING_TOL)

    def test_computed_torque_follows_pd_law(self):
        _assert_pd_computed_torque(self.result, self.joint_ids, kp=40.0, kd=5.0)

    def test_applied_torque_follows_max_effort_clamp(self):
        _assert_max_effort_clamp(self.result, self.joint_ids, effort_limit=80.0)


class TestDCMotorBehavior(unittest.TestCase):
    """DCMotor on all 12 joints: exact speed-torque clamp semantics under saturation."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            DC_MOTOR_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=DC_SATURATION_OFFSET,
            num_steps=20,
        )
        cls.joint_ids = _matching_joint_ids(cls.result, [".*HAA", ".*HFE", ".*KFE"])

    def test_computed_torque_follows_pd_law(self):
        _assert_pd_computed_torque(self.result, self.joint_ids, kp=40.0, kd=5.0)

    def test_applied_torque_follows_dc_motor_clamp(self):
        _assert_dc_motor_clamp(
            self.result, self.joint_ids, saturation_effort=120.0, velocity_limit=7.5, effort_limit=80.0
        )

    def test_saturation_engages(self):
        # kp * offset = 100 N*m > effort_limit = 80 N*m at step 0, so the clamp must actually cut torque.
        computed = self.result["computed_torque"][0][:, self.joint_ids]
        applied = self.result["applied_torque"][0][:, self.joint_ids]
        self.assertGreater(
            (computed - applied).abs().max().item(),
            1.0,
            "expected the DC-motor clamp to saturate at step 0",
        )

    def test_applied_torque_never_exceeds_effort_limit(self):
        for step_index, applied in enumerate(self.result["applied_torque"]):
            max_abs = applied[:, self.joint_ids].abs().max().item()
            self.assertLessEqual(
                max_abs, 80.0 + 1e-3, f"applied torque {max_abs:.3f} exceeds the effort limit at step {step_index}"
            )


class TestMixedActuatorBehavior(unittest.TestCase):
    """IdealPD on HAA + DCMotor on HFE/KFE: each group follows its own clamp law."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            MIXED_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=DC_SATURATION_OFFSET,
            num_steps=20,
        )
        cls.hip_ids = _matching_joint_ids(cls.result, ".*HAA")
        cls.knee_ids = _matching_joint_ids(cls.result, [".*HFE", ".*KFE"])

    def test_computed_torque_follows_pd_law(self):
        _assert_pd_computed_torque(self.result, self.hip_ids + self.knee_ids, kp=40.0, kd=5.0)

    def test_hip_group_follows_max_effort_clamp(self):
        _assert_max_effort_clamp(self.result, self.hip_ids, effort_limit=80.0)

    def test_knee_group_follows_dc_motor_clamp(self):
        _assert_dc_motor_clamp(
            self.result, self.knee_ids, saturation_effort=120.0, velocity_limit=7.5, effort_limit=80.0
        )


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


class TestMixedWithImplicitBehavior(unittest.TestCase):
    """Implicit HAA + IdealPD HFE + DCMotor KFE: all three actuation modes coexist.

    The implicit joints are driven by the solver's built-in joint drive
    (their sim gains are written; telemetry synthesizes the shadow PD),
    the explicit joints by Newton actuators. With gravity disabled every
    joint must reach its commanded pose regardless of actuation mode.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            MIXED_WITH_IMPLICIT_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=TRACKING_OFFSET,
            num_steps=TRACKING_STEPS,
        )
        cls.implicit_ids = _matching_joint_ids(cls.result, ".*HAA")
        cls.ideal_ids = _matching_joint_ids(cls.result, ".*HFE")
        cls.dc_ids = _matching_joint_ids(cls.result, ".*KFE")

    def test_explicit_joints_reach_commanded_pose(self):
        _assert_target_reached_and_held(
            self,
            self.result,
            hold_steps=TRACKING_HOLD_STEPS,
            tol=TRACKING_TOL,
            joint_ids=self.ideal_ids + self.dc_ids,
        )

    @_implicit_drive_inert_upstream
    def test_implicit_joints_reach_commanded_pose(self):
        _assert_target_reached_and_held(
            self,
            self.result,
            hold_steps=TRACKING_HOLD_STEPS,
            tol=TRACKING_TOL,
            joint_ids=self.implicit_ids,
        )

    def test_implicit_joints_report_drive_law_telemetry(self):
        _assert_implicit_drive_telemetry(self.result, self.implicit_ids)

    def test_ideal_pd_joints_follow_pd_law_and_box_clamp(self):
        _assert_pd_computed_torque(self.result, self.ideal_ids, kp=40.0, kd=5.0)
        _assert_max_effort_clamp(self.result, self.ideal_ids, effort_limit=80.0)

    def test_dc_motor_joints_follow_dc_clamp(self):
        _assert_pd_computed_torque(self.result, self.dc_ids, kp=40.0, kd=5.0)
        _assert_dc_motor_clamp(self.result, self.dc_ids, saturation_effort=120.0, velocity_limit=7.5, effort_limit=80.0)


# ---------------------------------------------------------------------------
# Implicit-only: no Newton actuators are built, the solver PD does the work
# ---------------------------------------------------------------------------

IMPLICIT_ONLY_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
    ),
}


class TestImplicitOnlyBehavior(unittest.TestCase):
    """All-implicit articulation: solver-internal PD drives, no adapter is built."""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            IMPLICIT_ONLY_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=TRACKING_OFFSET,
            num_steps=TRACKING_STEPS,
        )
        cls.joint_ids = _matching_joint_ids(cls.result, [".*HAA", ".*HFE", ".*KFE"])

    def test_no_newton_actuators_built(self):
        self.assertEqual(self.result["num_model_actuators"], 0)
        self.assertFalse(self.result["adapter_exists"])

    @_implicit_drive_inert_upstream
    def test_commanded_pose_reached_and_held(self):
        _assert_target_reached_and_held(self, self.result, hold_steps=TRACKING_HOLD_STEPS, tol=TRACKING_TOL)

    def test_telemetry_reports_drive_law(self):
        _assert_implicit_drive_telemetry(self.result, self.joint_ids)


# ---------------------------------------------------------------------------
# Implicit + non-zero feedforward effort target
# ---------------------------------------------------------------------------


class TestImplicitFeedforwardBehavior(unittest.TestCase):
    """Implicit-only actuators with a non-zero feedforward effort target.

    The FF lands additively on top of the solver's joint-drive PD, so with
    gravity disabled the equilibrium shifts to ``target + feedforward / kp``
    and the telemetry reports ``PD + FF``.
    """

    FEEDFORWARD = 2.0  # [N·m]
    KP = 40.0

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            IMPLICIT_ONLY_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=TRACKING_OFFSET,
            feedforward=cls.FEEDFORWARD,
            num_steps=TRACKING_STEPS,
        )
        cls.joint_ids = _matching_joint_ids(cls.result, [".*HAA", ".*HFE", ".*KFE"])

    @_implicit_drive_inert_upstream
    def test_steady_state_shifts_by_ff_over_kp(self):
        expected = self.result["target_pos"] + self.FEEDFORWARD / self.KP
        hold_mean = torch.stack(self.result["joint_pos"][-TRACKING_HOLD_STEPS:]).mean(dim=0)
        torch.testing.assert_close(
            hold_mean,
            expected,
            atol=0.02,
            rtol=0.0,
            msg="steady-state pose must shift by feedforward / kp on top of the commanded target",
        )

    def test_telemetry_includes_feedforward(self):
        _assert_implicit_drive_telemetry(self.result, self.joint_ids)


# ---------------------------------------------------------------------------
# Multi-articulation Newton scene (regression test for class-attr clobber)
# ---------------------------------------------------------------------------


CARTPOLE_EXPLICIT_ACTUATORS = {
    "all_joints": IdealPDActuatorCfg(
        joint_names_expr=["slider_to_cart", "cart_to_pole"],
        stiffness=100.0,
        damping=20.0,
        effort_limit=100.0,
    ),
}


def _run_anymal_and_cartpole(*, num_steps: int = TRACKING_STEPS) -> dict:
    """Spawn ANYmal-C + Cartpole per env (different DOF counts, different base types).

    Gravity is disabled and no ground plane is spawned so each robot's
    joint-space PD is the only torque source and both must settle exactly on
    their commanded targets.

    Returns:
        Mapping of robot name to a per-robot result dict shaped like
        :func:`_run_simulation`'s output (the subset the formula helpers use).
    """
    from isaaclab_assets import CARTPOLE_CFG  # noqa: PLC0415

    # Gravity must live on the explicit ``sim_cfg`` (see the note in _run_simulation).
    sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG, gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=False,
        add_ground_plane=False,
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

        robots = {"anymal": anymal, "cartpole": cartpole}
        results = {}
        for robot in robots.values():
            # Raw articulations have no env reset event: start from the asset's
            # default joint state (see the same block in _run_simulation).
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        for name, robot in robots.items():
            init_pos = wp.to_torch(robot.data.joint_pos).clone()
            results[name] = {
                "joint_names": tuple(robot.joint_names),
                "init_pos": init_pos,
                "init_vel": wp.to_torch(robot.data.joint_vel).clone(),
                "target_pos": init_pos + TRACKING_OFFSET,
                "target_vel": torch.zeros_like(init_pos),
                "effort_target": None,
                "joint_pos": [],
                "joint_vel": [],
                "computed_torque": [],
                "applied_torque": [],
            }
            robot.set_joint_position_target_index(target=results[name]["target_pos"])
            robot.set_joint_velocity_target_index(target=results[name]["target_vel"])

        for _ in range(num_steps):
            anymal.write_data_to_sim()
            cartpole.write_data_to_sim()
            sim.step()
            anymal.update(DT)
            cartpole.update(DT)
            for name, robot in robots.items():
                results[name]["joint_pos"].append(wp.to_torch(robot.data.joint_pos).clone())
                results[name]["joint_vel"].append(wp.to_torch(robot.data.joint_vel).clone())
                results[name]["computed_torque"].append(wp.to_torch(robot.data.computed_torque).clone())
                results[name]["applied_torque"].append(wp.to_torch(robot.data.applied_torque).clone())

    return results


class TestHeterogeneousMultiArticulationNewton(unittest.TestCase):
    """Two structurally-different articulations (ANYmal floating + Cartpole fixed) on Newton.

    Regression for the singleton-clobber bug in ``NewtonManager._adapter``
    / ``_post_actuator_callback``. Heterogeneous DOF counts (12 vs 2) and
    base types stress the global adapter's per-articulation DOF mapping: a
    clobbered mapping would compute one robot's torques from the other's
    state or targets, breaking the per-robot PD law and the tracking below.
    """

    @classmethod
    def setUpClass(cls):
        cls.results = _run_anymal_and_cartpole()

    def test_anymal_torques_follow_own_pd_law(self):
        result = self.results["anymal"]
        joint_ids = _matching_joint_ids(result, [".*HAA", ".*HFE", ".*KFE"])
        _assert_pd_computed_torque(result, joint_ids, kp=40.0, kd=5.0)
        _assert_max_effort_clamp(result, joint_ids, effort_limit=80.0)

    def test_cartpole_torques_follow_own_pd_law(self):
        result = self.results["cartpole"]
        joint_ids = _matching_joint_ids(result, ["slider_to_cart", "cart_to_pole"])
        _assert_pd_computed_torque(result, joint_ids, kp=100.0, kd=20.0)
        _assert_max_effort_clamp(result, joint_ids, effort_limit=100.0)

    def test_anymal_reaches_commanded_pose(self):
        _assert_target_reached_and_held(self, self.results["anymal"], hold_steps=TRACKING_HOLD_STEPS, tol=TRACKING_TOL)

    def test_cartpole_reaches_commanded_pose(self):
        _assert_target_reached_and_held(
            self, self.results["cartpole"], hold_steps=TRACKING_HOLD_STEPS, tol=TRACKING_TOL
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
        sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG)
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
            self.assertIsNotNone(adapter, "Newton adapter should exist for explicit actuator configs")
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

        sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG)
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


class TestNewtonActuatorGainSnapshotEnvStride(unittest.TestCase):
    """Regression: the init-time kp/kd snapshot must be correct for every env.

    ``build_newton_actuator_defaults`` scatters each Newton actuator's
    ``controller.kp`` / ``controller.kd`` into a per-articulation
    ``(num_envs, num_joints)`` tensor (``newton_default_stiffness`` /
    ``newton_default_damping``), which ``randomize_actuator_gains`` reads as
    its DR baseline. On a floating-base articulation the actuator ``indices``
    are laid out env-major with a per-env stride equal to the *whole model's*
    per-env DOF count (free-root DOFs + joints), which exceeds
    ``articulation.num_joints``. If the scatter decodes the env with
    ``num_joints`` instead of that stride, env 1's DOFs alias to the wrong
    rows (and partly out of bounds), corrupting the snapshot for every env
    past the first.

    ANYmal-C is floating base (6 free-root DOFs + 12 actuated joints -> a
    per-env stride of 18 vs. ``num_joints == 12``), so the bug manifests here
    with ``NUM_ENVS == 2``: without the fix, ``newton_default_stiffness[1]``
    is not uniformly the configured gain (its leading entries stay zero, as
    they are never written).
    """

    def test_snapshot_matches_config_for_all_envs(self):
        sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG)
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
            assert anymal.is_initialized

            stiffness = anymal.newton_default_stiffness
            damping = anymal.newton_default_damping
            self.assertIsNotNone(stiffness, "expected a Newton kp snapshot for explicit actuator groups")
            self.assertIsNotNone(damping, "expected a Newton kd snapshot for explicit actuator groups")

            n_j = anymal.num_joints
            self.assertEqual(tuple(stiffness.shape), (NUM_ENVS, n_j))
            self.assertEqual(tuple(damping.shape), (NUM_ENVS, n_j))

            # IDEAL_PD_ACTUATORS covers all 12 joints with constant gains, so
            # every cell of both env rows must equal the configured value.
            expected_kp = torch.full((NUM_ENVS, n_j), 40.0, device=anymal.device)
            expected_kd = torch.full((NUM_ENVS, n_j), 5.0, device=anymal.device)
            torch.testing.assert_close(stiffness, expected_kp)
            torch.testing.assert_close(damping, expected_kd)


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


class TestDelayedPDDelaySemantics(unittest.TestCase):
    """DelayedPDActuator: a target switch takes effect exactly ``delay_steps`` later.

    USD authoring maps ``DelayedPDActuatorCfg.max_delay`` to a fixed per-DOF
    ``delay_steps``, so the Newton input delay is deterministic. The test
    holds the initial pose target (the robot stays at rest with gravity
    disabled), switches the target at step ``SWITCH_STEP``, and recovers the
    actuator's *active* target from telemetry each step by inverting the PD
    law::

        target = (computed_torque + kd * vel) / kp + pos

    The recovered target must stay at the old value for exactly
    ``DELAY_STEPS`` steps after the switch and jump to the new value then.
    """

    KP = 40.0
    KD = 5.0
    DELAY_STEPS = 4  # authored from DELAYED_PD_ACTUATORS["legs"].max_delay
    SWITCH_STEP = 6
    RETARGET_OFFSET = 0.4  # [rad]

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            DELAYED_PD_ACTUATORS,
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=0.0,
            retarget_offset=cls.RETARGET_OFFSET,
            retarget_at_step=cls.SWITCH_STEP,
            num_steps=16,
        )
        cls.joint_ids = _matching_joint_ids(cls.result, [".*HAA", ".*HFE", ".*KFE"])

    def _recovered_target(self, step_index: int) -> torch.Tensor:
        pos_prev, vel_prev = _pre_step_states(self.result)
        computed = self.result["computed_torque"][step_index][:, self.joint_ids]
        return (computed + self.KD * vel_prev[step_index][:, self.joint_ids]) / self.KP + pos_prev[step_index][
            :, self.joint_ids
        ]

    def test_switch_is_masked_for_delay_steps(self):
        old_target = self.result["target_pos"][:, self.joint_ids]
        for step_index in range(self.SWITCH_STEP + self.DELAY_STEPS):
            torch.testing.assert_close(
                self._recovered_target(step_index),
                old_target,
                atol=1e-3,
                rtol=0.0,
                msg=f"actuator responded to the new target too early at step {step_index}",
            )

    def test_switch_takes_effect_after_delay_steps(self):
        new_target = self.result["target_pos2"][:, self.joint_ids]
        for step_index in range(self.SWITCH_STEP + self.DELAY_STEPS, len(self.result["computed_torque"])):
            torch.testing.assert_close(
                self._recovered_target(step_index),
                new_target,
                atol=1e-3,
                rtol=0.0,
                msg=f"actuator did not track the new target at step {step_index}",
            )


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
# Decimation folding: one manager step covering d sub-steps must match d
# explicit single sub-steps of the same path (self-consistency)
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


class _DecimationConsistencyBase(unittest.TestCase):
    """Folded-vs-stepped decimation self-consistency for one actuator config.

    Both runs drive the same Newton actuator path with identical commands;
    the only difference is whether the physics manager owns the decimation
    loop (a single folded ``step()`` covering ``decimation`` sub-steps,
    captured as one CUDA graph) or the test loops ``decimation`` single
    sub-steps itself. The reference is the same actuation path — this is
    self-consistency of the current implementation, not a comparison
    against a different one.
    """

    __test__ = False
    actuators: dict = {}
    dt: float = 1.0 / 100.0
    num_steps: int = 5
    decimation: int = 2
    pos_atol: float = 2e-3
    pos_rtol: float = 1e-3
    vel_atol: float = 1e-2
    vel_rtol: float = 1e-2
    torque_atol: float = 1e-3
    torque_rtol: float = 1e-3

    @classmethod
    def setUpClass(cls):
        kwargs = dict(
            dt=cls.dt,
            newton_cfg=NEWTON_CFG_DEC,
            num_steps=cls.num_steps,
            decimation=cls.decimation,
        )
        cls.folded_result = _run_simulation(cls.actuators, fold_decimation=True, **kwargs)
        cls.stepped_result = _run_simulation(cls.actuators, fold_decimation=False, **kwargs)

    def _assert_traces_match(self, field: str, atol: float, rtol: float) -> None:
        for step_index, (folded, stepped) in enumerate(
            zip(self.folded_result[field], self.stepped_result[field], strict=True)
        ):
            torch.testing.assert_close(
                folded,
                stepped,
                atol=atol,
                rtol=rtol,
                msg=f"{field} diverged between folded and stepped decimation at step {step_index}",
            )

    def test_manager_folds_decimation(self):
        self.assertTrue(self.folded_result["folds_decimation"], "expected the manager to fold the decimation loop")
        self.assertFalse(self.stepped_result["folds_decimation"])

    def test_joint_positions_match(self):
        self._assert_traces_match("joint_pos", self.pos_atol, self.pos_rtol)

    def test_joint_velocities_match(self):
        self._assert_traces_match("joint_vel", self.vel_atol, self.vel_rtol)

    def test_computed_torque_match(self):
        self._assert_traces_match("computed_torque", self.torque_atol, self.torque_rtol)

    def test_applied_torque_match(self):
        self._assert_traces_match("applied_torque", self.torque_atol, self.torque_rtol)


class TestDecimationIdealPD(_DecimationConsistencyBase):
    """IdealPD — folded decimation=2 + CUDA graph matches explicit sub-stepping."""

    __test__ = True
    actuators = IDEAL_PD_ACTUATORS


class TestDecimationDCMotor(_DecimationConsistencyBase):
    """DCMotor — folded decimation=2 + CUDA graph matches explicit sub-stepping."""

    __test__ = True
    actuators = DC_MOTOR_ACTUATORS


class TestDecimationDelayedPD(_DecimationConsistencyBase):
    """DelayedPD — the delay queue is stepped once per sub-step in both modes."""

    __test__ = True
    actuators = DELAYED_PD_ACTUATORS


class TestDecimationMixed(_DecimationConsistencyBase):
    """Mixed (IdealPD + DCMotor) — folded decimation=2 matches explicit sub-stepping."""

    __test__ = True
    actuators = MIXED_ACTUATORS


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
    """

    RESET_ENV: int = 0

    def _build_and_warm(self):
        sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG)
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
        ctx, sim, articulation = self._build_and_warm()
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
            dof_env_id = adapter._dof_env_id.numpy()
            for act, state in stateful_pairs:
                pushes_after = state.delay_state.num_pushes.numpy()
                indices_np = act.indices.numpy()
                for i, global_dof in enumerate(indices_np):
                    env = int(dof_env_id[int(global_dof)])
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
    sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG)

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


def _remotized_actuators() -> dict:
    """IdealPD hips + RemotizedPD knees (Spot lookup table on the KFE joints)."""
    from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg  # noqa: PLC0415

    return {
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


class TestRemotizedPDBehavior(unittest.TestCase):
    """RemotizedPD (PD + delay + position-based clamping): exact clamp law on the knees.

    Targets are constant, so the authored input delay is transparent and the
    PD law holds on every step; the knee ``applied_torque`` must equal
    ``computed_torque`` clamped to ``±min(effort_limit, interp(pos))`` with
    the Spot lookup table, matching Newton's ``ClampingPositionBased``
    boundary-clamped linear interpolation.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _run_simulation(
            _remotized_actuators(),
            gravity_enabled=False,
            add_ground_plane=False,
            target_offset=0.4,
            num_steps=20,
        )
        cls.hip_ids = _matching_joint_ids(cls.result, [".*HAA", ".*HFE"])
        cls.knee_ids = _matching_joint_ids(cls.result, ".*KFE")

    def test_hips_follow_pd_law_and_box_clamp(self):
        _assert_pd_computed_torque(self.result, self.hip_ids, kp=40.0, kd=5.0)
        _assert_max_effort_clamp(self.result, self.hip_ids, effort_limit=80.0)

    def test_knees_follow_pd_law(self):
        _assert_pd_computed_torque(self.result, self.knee_ids, kp=60.0, kd=1.5)

    def test_knees_follow_position_based_clamp(self):
        lookup = np.asarray(SPOT_KNEE_LOOKUP, dtype=np.float32)
        pos_prev, _ = _pre_step_states(self.result)
        for step_index, (computed, applied) in enumerate(
            zip(self.result["computed_torque"], self.result["applied_torque"])
        ):
            knee_pos = pos_prev[step_index][:, self.knee_ids].cpu().numpy()
            limit_np = np.interp(knee_pos.ravel(), lookup[:, 0], lookup[:, 2]).reshape(knee_pos.shape)
            limit = torch.from_numpy(np.minimum(limit_np, 80.0).astype(np.float32)).to(computed.device)
            expected = torch.clamp(computed[:, self.knee_ids], min=-limit, max=limit)
            torch.testing.assert_close(
                applied[:, self.knee_ids],
                expected,
                atol=1e-3,
                rtol=1e-4,
                msg=f"applied_torque broke the position-based clamp at step {step_index}",
            )


class TestDecimationRemotizedPD(_DecimationConsistencyBase):
    """RemotizedPD — folded decimation=2 + CUDA graph matches explicit sub-stepping."""

    __test__ = True

    @classmethod
    def setUpClass(cls):
        cls.actuators = _remotized_actuators()
        super().setUpClass()


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
                "Expected Newton model actuators to be non-empty on the Newton backend.",
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


def test_newton_actuator_defaults_follow_requested_public_joint_order() -> None:
    """Convert Newton actuator gain snapshots and managed IDs into public joint order.

    ``build_newton_actuator_defaults`` gathers each articulation's gains from the
    adapter's flat global-DOF snapshots (an actuator with DOFs {0, 2} in env 0 and
    {3, 5} in env 1, kp 10/30 and 11/31) through the DOF index map, then permutes
    into public joint order.
    """
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    kp_flat = torch.tensor([10.0, 0.0, 30.0, 11.0, 0.0, 31.0])
    kd_flat = torch.tensor([1.0, 0.0, 3.0, 1.1, 0.0, 3.1])
    managed_flat = torch.tensor([True, False, True, True, False, True])
    dof_index_map = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)

    stiffness, damping, managed = build_newton_actuator_defaults(
        kp_flat=kp_flat,
        kd_flat=kd_flat,
        managed_flat=managed_flat,
        dof_index_map=dof_index_map,
        joint_user_to_backend_indices=(2, 0, 1),
    )

    torch.testing.assert_close(stiffness, torch.tensor([[30.0, 10.0, 0.0], [31.0, 11.0, 0.0]]))
    torch.testing.assert_close(damping, torch.tensor([[3.0, 1.0, 0.0], [3.1, 1.1, 0.0]]))
    torch.testing.assert_close(managed, torch.tensor([0, 1], dtype=torch.int32))


def test_newton_actuator_defaults_reject_incomplete_joint_permutation() -> None:
    """Reject malformed actuator-default ordering maps with an actionable error."""
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    with pytest.raises(
        ValueError,
        match=(
            r"joint_user_to_backend_indices must contain each backend joint index exactly once; "
            r"expected a permutation of 0\.\.2, got \(0, 0, 2\)\."
        ),
    ):
        build_newton_actuator_defaults(
            kp_flat=torch.zeros(3),
            kd_flat=torch.zeros(3),
            managed_flat=torch.zeros(3, dtype=torch.bool),
            dof_index_map=torch.arange(3, dtype=torch.long).reshape(1, 3),
            joint_user_to_backend_indices=(0, 0, 2),
        )


if __name__ == "__main__":
    unittest.main()
