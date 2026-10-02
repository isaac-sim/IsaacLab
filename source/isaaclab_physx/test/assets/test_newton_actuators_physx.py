# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-native actuator seams on the PhysX backend.

One rollout steps every robot with Isaac Lab actuators; a second rollout steps the same robots, plus a few that
only exist on the Newton path, with Newton-native actuators (authored from the same Lab configs and stepped through
:class:`PhysxActuatorWrapper`). The translation of each actuator model is backend independent and owned by the
Newton backend suite; these tests keep the PhysX seams: wrapper efforts next to implicit PhysX drives, stateful
actuators with ping-pong CUDA graphs, non-identity joint ordering, per-articulation adapters on heterogeneous
robots, per-environment resets, gain randomization, and network actuators.
"""

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices
from isaaclab.utils import replace

launch_test_simulation()

import os
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.test.utils.actuator_equivalence import (
    CARTPOLE_EXPLICIT_ACTUATORS,
    DELAYED_PD_ACTUATORS,
    IDEAL_PD_ACTUATORS,
    IMPLICIT_ONLY_ACTUATORS,
    MIXED_WITH_IMPLICIT_ACTUATORS,
    MockEnv,
    build_dr_term,
    make_dummy_lstm_checkpoint,
    make_dummy_mlp_checkpoint,
)
from isaaclab.test.utils.articulation_ordering import assert_articulation_ordering_trace_matches

from isaaclab_assets import ANYMAL_C_CFG, CARTPOLE_CFG

NUM_ENVS = 2
NUM_STEPS = 10
DT = 1.0 / 120.0
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
# Absolute and relative tolerance of each trace in the Lab-versus-Newton comparison.
_TRACE_TOLERANCES = {
    "joint_pos": (2e-3, 1e-3),
    "joint_vel": (1e-2, 1e-2),
    "computed_effort": (1e-3, 1e-3),
    "applied_effort": (1e-3, 1e-3),
}


_ROWS = {"delayed": 0, "mixed": 1, "ideal": 2, "reversed": 2, "cartpole": 3, "implicit": 4, "mlp": 5, "lstm": 6}
"""Row of each robot along y. Robots compared across rollouts share a row: ground contacts amplify the floating-point
differences between placements at different world positions."""
_LAB_ROBOTS = ("delayed", "mixed", "ideal", "cartpole")
"""Robots of the Lab-actuator rollout, which the Newton-actuator rollout reproduces."""
_NEWTON_ROBOTS = (*_LAB_ROBOTS, "implicit", "mlp", "lstm")
"""Robots of the Newton-actuator rollout."""


def _robot_cfgs(names: tuple[str, ...], networks: dict[str, str]) -> dict[str, ArticulationCfg]:
    """Return the configurations of the named robots.

    Args:
        names: Robot names. ``reversed`` is the ``ideal`` robot with a reversed public joint order.
        networks: Paths of the ``mlp`` and ``lstm`` actuator checkpoints.
    """
    network_limits = {"saturation_effort": 120.0, "actuator_effort_limit": 80.0, "actuator_velocity_limit": 7.5}
    pd_legs = IdealPDActuatorCfg(
        joint_names_expr=[".*HFE", ".*KFE"], stiffness=40.0, damping=5.0, actuator_effort_limit=80.0
    )
    anymal_actuators = {
        "delayed": DELAYED_PD_ACTUATORS,
        "mixed": MIXED_WITH_IMPLICIT_ACTUATORS,
        "ideal": IDEAL_PD_ACTUATORS,
        "implicit": IMPLICIT_ONLY_ACTUATORS,
        "mlp": {
            "mlp_legs": ActuatorNetMLPCfg(
                joint_names_expr=[".*HAA"],
                network_file=networks["mlp"],
                pos_scale=-1.0,
                vel_scale=1.0,
                torque_scale=1.0,
                input_order="pos_vel",
                input_idx=[0, 1, 2],
                **network_limits,
            ),
            "pd_legs": pd_legs,
        },
        "lstm": {
            "lstm_legs": ActuatorNetLSTMCfg(
                joint_names_expr=[".*HAA"], network_file=networks["lstm"], **network_limits
            ),
            "pd_legs": pd_legs,
        },
    }
    cfgs = {}
    for name in names:
        if name == "cartpole":
            cfgs[name] = replace(CARTPOLE_CFG, actuators=CARTPOLE_EXPLICIT_ACTUATORS)
        elif name == "reversed":
            joint_ordering = tuple(reversed(_ANYMAL_C_PHYSX_JOINT_NAMES))
            cfgs[name] = replace(ANYMAL_C_CFG, actuators=IDEAL_PD_ACTUATORS, joint_ordering=joint_ordering)
        else:
            cfgs[name] = replace(ANYMAL_C_CFG, actuators=anymal_actuators[name])
    return cfgs


def _rollout(sim: SimulationContext, cfgs: dict[str, ArticulationCfg]) -> tuple[dict[str, Articulation], dict]:
    """Spawn the robots side by side, command permutation-sensitive targets, and record ``NUM_STEPS`` steps.

    Args:
        sim: Simulation context the robots are spawned in.
        cfgs: Robot configurations keyed by name.

    Returns:
        The articulations and their recorded traces, both keyed by robot name.
    """
    for env_index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(3.0 * env_index, 0.0, 0.0))
    articulations = {
        name: Articulation(
            replace(
                cfg,
                prim_path=f"/World/Env_[^/]*/{name}",
                init_state=replace(cfg.init_state, pos=(0.0, 3.0 * _ROWS[name], cfg.init_state.pos[2])),
            )
        )
        for name, cfg in cfgs.items()
    }
    sim.reset()

    traces = {}
    for name, articulation in articulations.items():
        assert articulation.is_initialized, name
        joint_names = tuple(articulation.joint_names)
        backend_joint_names = tuple(articulation.backend_joint_names)
        ordering = articulation.joint_ordering
        # Distinct commands per physical joint, so that a joint-order mix-up cannot pass.
        scale = torch.tensor([backend_joint_names.index(joint) + 1.0 for joint in joint_names], device=sim.device)
        scale = scale.expand(NUM_ENVS, -1)
        target_pos = articulation.data.default_joint_pos.torch + 0.01 * scale
        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=0.001 * scale)
        articulation.set_joint_effort_target_index(target=0.1 * scale)
        traces[name] = {
            "joint_names": joint_names,
            "backend_joint_names": backend_joint_names,
            "adapter_joint_names": joint_names,
            "joint_ordering": None
            if ordering is None
            else {
                "user_names": joint_names,
                "backend_names": backend_joint_names,
                "user_to_backend_indices": ordering.user_to_backend_indices,
                "backend_to_user_indices": ordering.backend_to_user_indices,
            },
            "user_to_backend": [backend_joint_names.index(joint) for joint in joint_names],
            "target_pos": target_pos,
            "target_vel": 0.001 * scale,
            "effort_target": 0.1 * scale,
            "raw_joint_pos": [],
            "raw_joint_vel": [],
            "adapter_applied_effort": [],
            **{field: [] for field in _TRACE_TOLERANCES},
        }

    for _ in range(NUM_STEPS):
        for name, articulation in articulations.items():
            trace = traces[name]
            # Read the raw PhysX state first: it bypasses the data's joint-state shadow, so it cannot refresh a stale
            # shadow before the actuators read it.
            for key, read in (("raw_joint_pos", "get_dof_positions"), ("raw_joint_vel", "get_dof_velocities")):
                trace[key].append(wp.to_torch(getattr(articulation.root_view, read)())[:, trace["user_to_backend"]])
            articulation.write_data_to_sim()
            # Record the state the actuators acted on and their outputs. Reading the public state only after the
            # actuators ran keeps a stale shadow observable on the next step.
            trace["joint_pos"].append(articulation.data.joint_pos.torch.clone())
            trace["joint_vel"].append(articulation.data.joint_vel.torch.clone())
            trace["computed_effort"].append(articulation.actuators.computed_effort.torch.clone())
            trace["applied_effort"].append(articulation.actuators.applied_effort.torch.clone())
            if articulation._physx_actuator_wrapper is not None:
                wrapper_effort = wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d)
                trace["adapter_applied_effort"].append(wrapper_effort.clone())
        sim.step()
        for articulation in articulations.values():
            articulation.update(DT)
    return articulations, traces


@dataclass
class _Rollouts:
    """Recorded traces of the three rollouts and the live robots of the Newton-actuator rollout."""

    lab: dict[str, dict]
    """Traces of the Lab-actuator rollout."""
    newton: dict[str, dict]
    """Traces of the Newton-actuator rollout."""
    reversed: dict
    """Trace of the ``reversed`` robot, recorded alone with Newton-native actuators."""
    robots: dict[str, Articulation]
    """Robots of the Newton-actuator rollout, whose simulation stays alive for the tests."""


def _simulation(device: str, use_newton_actuators: bool):
    """Return a simulation context with gravity and a ground plane on the PhysX backend."""
    sim_cfg = SimulationCfg(dt=DT, physics=PhysxCfg(), use_newton_actuators=use_newton_actuators)
    return build_simulation_context(device=device, gravity_enabled=True, add_ground_plane=True, sim_cfg=sim_cfg)


@pytest.fixture(scope="module", params=test_devices(DeviceScope.DEFAULT_CUDA))
def rollouts(request) -> Iterator[_Rollouts]:
    """Record the Lab-actuator and reversed-ordering rollouts, then keep the Newton-actuator rollout alive."""
    device = request.param
    networks = {"mlp": make_dummy_mlp_checkpoint(), "lstm": make_dummy_lstm_checkpoint()}
    try:
        traces = {}
        for name, use_newton_actuators, names in (("lab", False, _LAB_ROBOTS), ("reversed", True, ("reversed",))):
            with _simulation(device, use_newton_actuators) as sim:
                sim._app_control_on_stop_handle = None
                traces[name] = _rollout(sim, _robot_cfgs(names, networks))[1]
        with _simulation(device, use_newton_actuators=True) as sim:
            sim._app_control_on_stop_handle = None
            robots, newton_traces = _rollout(sim, _robot_cfgs(_NEWTON_ROBOTS, networks))
            yield _Rollouts(traces["lab"], newton_traces, traces["reversed"]["reversed"], robots)
    finally:
        for path in networks.values():
            os.unlink(path)


@pytest.mark.parametrize("robot", ["delayed", "mixed", "ideal", "cartpole"])
def test_newton_actuators_match_lab_actuators(rollouts: _Rollouts, robot: str) -> None:
    """Newton-native actuators reproduce the Lab trajectories and efforts next to implicit PhysX drives."""
    for field, (atol, rtol) in _TRACE_TOLERANCES.items():
        for step, (lab, newton) in enumerate(
            zip(rollouts.lab[robot][field], rollouts.newton[robot][field], strict=True)
        ):
            torch.testing.assert_close(
                newton, lab, atol=atol, rtol=rtol, msg=f"{robot} {field} diverged at step {step}"
            )


def test_stateful_actuators_capture_ping_pong_graphs(rollouts: _Rollouts) -> None:
    """Graphable stateful Newton actuators capture the two ping-pong CUDA graphs."""
    assert len(rollouts.robots["delayed"]._actuator_control._native_actuator_graphs) == 2


def test_reversed_joint_ordering_matches_identity_ordering(rollouts: _Rollouts) -> None:
    """A reversed public joint order commands and observes the same physical joints as the backend order."""
    assert_articulation_ordering_trace_matches(
        rollouts.newton["ideal"], rollouts.reversed, tuple(reversed(_ANYMAL_C_PHYSX_JOINT_NAMES))
    )


def test_reversed_joint_ordering_uses_current_joint_state(rollouts: _Rollouts) -> None:
    """Regression: under a reversed joint order, each step's effort follows the IdealPD law on that step's state."""
    trace = rollouts.reversed
    kp, kd, effort_limit = 40.0, 5.0, 80.0
    for step, (applied, position, velocity) in enumerate(
        zip(trace["applied_effort"], trace["raw_joint_pos"], trace["raw_joint_vel"], strict=True)
    ):
        demand = kp * (trace["target_pos"] - position) + kd * (trace["target_vel"] - velocity) + trace["effort_target"]
        expected = demand.clamp(-effort_limit, effort_limit)
        torch.testing.assert_close(applied, expected, atol=1e-3, rtol=1e-3, msg=f"stale joint state at step {step}")


@pytest.mark.parametrize("robot", ["mlp", "lstm"])
def test_network_actuators_drive_haa_joints(rollouts: _Rollouts, robot: str) -> None:
    """Network actuators produce finite, non-zero HAA efforts that follow the moving joint state."""
    trace = rollouts.newton[robot]
    haa_ids = [index for index, name in enumerate(trace["joint_names"]) if name.endswith("HAA")]
    assert len(haa_ids) == 4
    for step, (position, effort) in enumerate(zip(trace["joint_pos"], trace["applied_effort"])):
        assert torch.isfinite(position).all(), f"non-finite positions at step {step}"
        assert torch.all(effort[:, haa_ids] != 0.0), f"zero HAA effort at step {step}"
    # A constant output (e.g. only the output-layer bias) would mean the network is not fed the joint state.
    assert not torch.allclose(trace["applied_effort"][0][:, haa_ids], trace["applied_effort"][-1][:, haa_ids])


def test_reset_clears_actuator_state_of_selected_envs(rollouts: _Rollouts) -> None:
    """Resetting environment 0 clears the delay history of its DoFs only."""
    articulation = rollouts.robots["delayed"]
    adapter = articulation.newton_actuator_adapter
    delay_states = [
        (actuator, state.delay_state)
        for actuator, state in zip(adapter.actuators, adapter._states_a)
        if state is not None and getattr(state, "delay_state", None) is not None
    ]
    assert delay_states
    assert all((delay_state.num_pushes.numpy() > 0).all() for _, delay_state in delay_states)

    articulation.reset(env_ids=torch.tensor([0], device=articulation.device))
    for actuator, delay_state in delay_states:
        env_of_dof = actuator.indices.numpy() // adapter.num_joints
        pushes = delay_state.num_pushes.numpy()
        assert (pushes[env_of_dof == 0] == 0).all()
        assert (pushes[env_of_dof != 0] > 0).all()


def test_gain_randomization_reuses_payload_for_implicit_storage(rollouts: _Rollouts) -> None:
    """Randomized implicit gains reach the actuator and the solver as one payload, in the selected env only."""
    robot = rollouts.robots["implicit"]
    actuator = robot.actuators["legs"]
    stiffness_before, damping_before = actuator.stiffness.clone(), actuator.damping.clone()
    env = MockEnv({"robot": robot}, NUM_ENVS, robot.device)
    term, asset_cfg = build_dr_term(env, "robot")
    env_ids = torch.tensor([0], device=robot.device)
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

    assert torch.unique(actuator.stiffness[env_ids]).numel() > 1
    assert torch.unique(actuator.damping[env_ids]).numel() > 1
    torch.testing.assert_close(actuator.stiffness[env_ids], robot.data.joint_stiffness.torch[env_ids])
    torch.testing.assert_close(actuator.damping[env_ids], robot.data.joint_damping.torch[env_ids])
    torch.testing.assert_close(actuator.stiffness[1:], stiffness_before[1:])
    torch.testing.assert_close(actuator.damping[1:], damping_before[1:])


def test_gain_randomization_reaches_only_the_selected_articulation(rollouts: _Rollouts) -> None:
    """Each PhysX articulation owns its adapter, so randomizing the cartpole leaves the ANYmal controllers alone."""
    anymal, cartpole = rollouts.robots["ideal"], rollouts.robots["cartpole"]
    assert anymal.newton_actuator_adapter is not None and cartpole.newton_actuator_adapter is not None
    assert anymal.newton_actuator_adapter is not cartpole.newton_actuator_adapter
    groups = {"anymal": (anymal, "legs"), "cartpole": (cartpole, "all_joints")}

    def read_gains() -> dict[tuple[str, str], torch.Tensor]:
        return {
            (name, gain): read_group_parameter(articulation.actuators, group, "controller", gain).clone()
            for name, (articulation, group) in groups.items()
            for gain in ("kp", "kd")
        }

    expected = read_gains()
    env = MockEnv({"anymal": anymal, "cartpole": cartpole}, NUM_ENVS, anymal.device)
    term, asset_cfg = build_dr_term(env, "cartpole")

    term(
        env,
        env_ids=torch.tensor([0], device=anymal.device),
        asset_cfg=asset_cfg,
        stiffness_distribution_params=(100.0, 100.0),
        damping_distribution_params=(5.0, 5.0),
        operation="abs",
        distribution="uniform",
    )

    expected["cartpole", "kp"][0] = 100.0
    expected["cartpole", "kd"][0] = 5.0
    for key, gains in read_gains().items():
        torch.testing.assert_close(gains, expected[key], msg=str(key))
