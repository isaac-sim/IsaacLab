# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Lab-versus-Newton actuator equivalence on locally authored articulations.

Isaac Lab actuators and Newton-native actuators (created from the same Lab configs via USD authoring) must
produce identical joint trajectories and torque telemetry. Each actuator configuration drives its own
two-environment island in one composite scene per execution path, so the Lab and the Newton paths each build
one scene. The legged island is floating based, which exercises the coordinate-vs-DOF index separation that
free joints introduce between ``joint_q`` and ``joint_qd``, and the cartpole island shares the Newton path's
model-wide actuator adapter with articulations of a different DOF count and base type.

The Newton scene stays alive for the tests that drive native actuators directly: gain writes and domain
randomization, target submission, and per-environment state reset.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.actuators import ActuatorBaseCfg, DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from isaaclab.actuators.actuator_pd_cfg import RemotizedPDActuatorCfg
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.test.utils.actuator_equivalence import (
    CARTPOLE_EXPLICIT_ACTUATORS,
    IDEAL_PD_ACTUATORS,
    IMPLICIT_ONLY_ACTUATORS,
    MIXED_WITH_IMPLICIT_ACTUATORS,
    EquivalenceAssertionsMixin,
    MockEnv,
    build_dr_term,
    make_dummy_lstm_checkpoint,
    make_dummy_mlp_checkpoint,
)
from isaaclab.test.utils.articulation_ordering import assert_articulation_ordering_trace_matches
from isaaclab.utils import replace

from isaaclab_assets.robots.spot import joint_parameter_lookup as SPOT_KNEE_LOOKUP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from articulation_test_utils import NUM_ENVS, local_usd, newton_sim_cfg, spawn_assets  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_STEPS = 10
DT = 1.0 / 120.0
TARGET_OFFSET = 0.1  # [rad] added to initial joint positions


def _solver_cfg() -> MJWarpSolverCfg:
    """Return the MJWarp solver configuration shared by the actuator scenes."""
    return MJWarpSolverCfg(
        njmax=500,
        nconmax=500,
        ls_iterations=20,
        cone="pyramidal",
        impratio=1,
        integrator="implicitfast",
    )


_LEG_REORDERED_JOINT_NAMES = ("LF_KFE", "RH_HAA", "LF_HAA", "RH_KFE", "LF_HFE", "RH_HFE")
"""A public order that permutes ``floating_two_leg.usda``'s backend joints without being its own inverse."""

_LEG_INIT_STATE = ArticulationCfg.InitialStateCfg(joint_pos={".*HAA": 0.0, ".*HFE": 0.4, ".*KFE": -0.8})
"""A bent-knee pose inside the Spot knee lookup table."""

# The PD demand (~kp * TARGET_OFFSET = 4 N·m) exceeds the 2 N·m saturation effort, so the DC-motor
# torque-speed clamp binds and Newton must author it for the paths to match.
SATURATING_DC_MOTOR_ACTUATORS = {
    "legs": DCMotorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        saturation_effort=2.0,
        actuator_effort_limit=80.0,
        actuator_velocity_limit=7.5,
        stiffness=40.0,
        damping=5.0,
    ),
}

# Newton authors ``max_delay`` as a fixed delay, while the Lab path samples a lag in
# ``[min_delay, max_delay]`` on reset; a fixed delay makes both paths apply the same lag.
FIXED_DELAYED_PD_ACTUATORS = {
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
        actuator_effort_limit=80.0,
        min_delay=2,
        max_delay=2,
    ),
}

# RemotizedPD (Spot knee lookup) on KFE with IdealPD on HAA/HFE.
REMOTIZED_PD_ACTUATORS = {
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
        min_delay=3,
        max_delay=3,
        joint_parameter_lookup=SPOT_KNEE_LOOKUP,
    ),
}


def _neural_actuators(mlp_path: str, lstm_path: str) -> dict[str, ActuatorBaseCfg]:
    """MLP on HAA, LSTM on HFE, and IdealPD on KFE."""
    return {
        "mlp_legs": ActuatorNetMLPCfg(
            joint_names_expr=[".*HAA"],
            network_file=mlp_path,
            saturation_effort=120.0,
            actuator_effort_limit=80.0,
            actuator_velocity_limit=7.5,
            pos_scale=-1.0,
            vel_scale=1.0,
            torque_scale=1.0,
            input_order="pos_vel",
            input_idx=[0, 1, 2],
        ),
        "lstm_legs": ActuatorNetLSTMCfg(
            joint_names_expr=[".*HFE"],
            network_file=lstm_path,
            saturation_effort=120.0,
            actuator_effort_limit=80.0,
            actuator_velocity_limit=7.5,
        ),
        "pd_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*KFE"],
            stiffness=40.0,
            damping=5.0,
            actuator_effort_limit=80.0,
        ),
    }


@dataclass
class _Island:
    """One actuator configuration under test and the commands its rollout sends."""

    actuators: dict[str, ActuatorBaseCfg]
    usd: str = "floating_two_leg.usda"
    joint_ordering: tuple[str, ...] | None = None
    feedforward: float | None = None
    """Constant per-DOF feedforward effort target, when not None."""
    ramp_targets: bool = False
    """Whether to ramp the position target over the rollout so command delays are observable."""
    permutation_sensitive_commands: bool = False
    """Whether to command distinct position, velocity, and effort values by physical joint name."""
    newton_only: bool = False
    """Whether only the Newton path runs the island."""
    torque_atol: float = EquivalenceAssertionsMixin.torque_atol
    """Absolute tolerance of the torque-telemetry oracles [N·m]."""


def _islands(mlp_path: str | None = None, lstm_path: str | None = None) -> dict[str, _Island]:
    """Return the actuator islands, with the neural island when the network checkpoints are given."""
    islands = {
        "ideal": _Island(IDEAL_PD_ACTUATORS),
        "ideal_reordered": _Island(IDEAL_PD_ACTUATORS, joint_ordering=_LEG_REORDERED_JOINT_NAMES),
        "cartpole": _Island(CARTPOLE_EXPLICIT_ACTUATORS, usd="fixed_cartpole.usda"),
        "dc_motor": _Island(SATURATING_DC_MOTOR_ACTUATORS),
        "mixed": _Island(MIXED_WITH_IMPLICIT_ACTUATORS),
        "implicit_feedforward": _Island(IMPLICIT_ONLY_ACTUATORS, feedforward=2.0, torque_atol=0.5),
        "delayed": _Island(FIXED_DELAYED_PD_ACTUATORS, ramp_targets=True),
        "remotized": _Island(REMOTIZED_PD_ACTUATORS, ramp_targets=True),
        # The implicit hip group reads its torque telemetry from backend-order effort buffers, so the
        # permutation-sensitive feedforward effort also checks that gather.
        "mixed_permuted": _Island(MIXED_WITH_IMPLICIT_ACTUATORS, permutation_sensitive_commands=True, newton_only=True),
        "mixed_permuted_reordered": _Island(
            MIXED_WITH_IMPLICIT_ACTUATORS,
            joint_ordering=_LEG_REORDERED_JOINT_NAMES,
            permutation_sensitive_commands=True,
            newton_only=True,
        ),
    }
    if mlp_path is not None:
        islands["neural"] = _Island(_neural_actuators(mlp_path, lstm_path), newton_only=True)
    return islands


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    """Articulations of one execution path and their recorded rollouts."""

    sim: SimulationContext
    articulations: dict[str, Articulation]
    results: dict[str, dict]


def _island_cfg(name: str, island: _Island, index: int) -> ArticulationCfg:
    """Return the articulation of one island, offset along y by its index."""
    init_state = ArticulationCfg.InitialStateCfg() if island.usd != "floating_two_leg.usda" else _LEG_INIT_STATE
    init_state = replace(init_state, pos=(0.0, 3.0 * index, 1.0))
    return ArticulationCfg(
        prim_path=f"/World/Env_[^/]*/{name}",
        spawn=local_usd(island.usd),
        init_state=init_state,
        actuators=island.actuators,
        joint_ordering=island.joint_ordering,
    )


def _run(
    sim: SimulationContext,
    islands: dict[str, _Island],
    use_newton_actuators: bool,
    *,
    dt: float = DT,
    num_steps: int = NUM_STEPS,
    decimation: int = 1,
) -> _Run:
    """Spawn one island per actuator configuration, roll every island out, and record trajectories and telemetry.

    Records ``joint_pos``, ``joint_vel``, ``computed_effort``, and ``applied_effort`` per step, the
    backend-order adapter efforts on the Newton path, joint-name metadata, the commands, and the Newton actuators
    that execute each group.
    """
    articulations = spawn_assets(
        {name: _island_cfg(name, island, index) for index, (name, island) in enumerate(islands.items())}
    )
    sim.reset()
    for articulation in articulations.values():
        assert articulation.is_initialized
        # Start from the configured joint state, as an environment reset would.
        articulation.write_joint_state_to_sim_index(
            position=articulation.data.default_joint_pos.torch.clone(),
            velocity=articulation.data.default_joint_vel.torch.clone(),
        )
        # Reset samples the Lab-path actuator delay; without it the Lab lag stays at zero.
        articulation.reset()

    if use_newton_actuators and decimation > 1:
        SimulationManager.set_decimation(decimation)
    handles_dec = (
        use_newton_actuators
        and decimation > 1
        and SimulationManager._is_all_graphable()
        and SimulationManager._decimation > 1
    )

    results = {}
    for name, island in islands.items():
        articulation = articulations[name]
        joint_names = tuple(articulation.joint_names)
        backend_joint_names = tuple(articulation.backend_joint_names)
        installed_ordering = articulation.joint_ordering
        init_pos = articulation.data.joint_pos.torch.clone()
        if island.permutation_sensitive_commands:
            scale_by_name = {joint_name: index + 1 for index, joint_name in enumerate(backend_joint_names)}
            joint_scale = torch.tensor(
                [scale_by_name[joint_name] for joint_name in joint_names], device=sim.device, dtype=init_pos.dtype
            ).expand_as(init_pos)
            target_pos = init_pos + 0.01 * joint_scale
            target_vel = 0.001 * joint_scale
            effort_target = 0.1 * joint_scale
        else:
            target_pos = init_pos + TARGET_OFFSET
            target_vel = torch.zeros_like(init_pos)
            effort_target = None if island.feedforward is None else torch.full_like(init_pos, island.feedforward)
        commands = articulation.actuators.target_command
        commands.set_position_index(value=target_pos)
        commands.set_velocity_index(value=target_vel)
        if effort_target is not None:
            commands.set_effort_index(value=effort_target)
        results[name] = {
            "joint_names": joint_names,
            "backend_joint_names": backend_joint_names,
            "joint_ordering": (
                None
                if installed_ordering is None
                else {
                    "user_names": joint_names,
                    "backend_names": backend_joint_names,
                    "user_to_backend_indices": installed_ordering.user_to_backend_indices,
                    "backend_to_user_indices": installed_ordering.backend_to_user_indices,
                    "is_identity": False,
                }
            ),
            "adapter_joint_names": backend_joint_names,
            "init_pos": init_pos,
            "target_pos": target_pos.clone(),
            "target_vel": target_vel.clone(),
            "effort_target": None if effort_target is None else effort_target.clone(),
            "joint_pos": [],
            "joint_vel": [],
            "computed_effort": [],
            "applied_effort": [],
            "adapter_applied_effort": [],
        }

    for step in range(num_steps):
        for name, island in islands.items():
            if island.ramp_targets:
                target_pos = results[name]["init_pos"] + TARGET_OFFSET * (step + 1) / num_steps
                articulations[name].actuators.target_command.set_position_index(value=target_pos)
                results[name]["target_pos"] = target_pos.clone()
        for _ in range(1 if handles_dec else decimation):
            for articulation in articulations.values():
                articulation.write_data_to_sim()
            sim.step()
            for articulation in articulations.values():
                articulation.update(dt * decimation if handles_dec else dt)
        for name, articulation in articulations.items():
            result = results[name]
            result["joint_pos"].append(articulation.data.joint_pos.torch.clone())
            result["joint_vel"].append(articulation.data.joint_vel.torch.clone())
            result["computed_effort"].append(articulation.actuators.computed_effort.torch.clone())
            result["applied_effort"].append(articulation.actuators.applied_effort.torch.clone())
            if use_newton_actuators:
                result["adapter_applied_effort"].append(wp.to_torch(articulation.data._sim_bind_joint_effort).clone())

    for name, articulation in articulations.items():
        actuator_info = []
        if use_newton_actuators:
            for group_name in islands[name].actuators:
                if group_name not in articulation.actuators._native_group_names:
                    continue
                group_actuators = articulation.actuators[group_name]
                if not isinstance(group_actuators, tuple):
                    group_actuators = (group_actuators,)
                for act in group_actuators:
                    actuator_info.append(
                        {
                            "group": group_name,
                            "controller_type": type(act.controller).__name__,
                            "clamping_types": sorted(type(c).__name__ for c in (act.clamping or [])),
                            "has_delay": act.delay is not None,
                        }
                    )
        results[name]["actuator_info"] = actuator_info
    return _Run(sim=sim, articulations=articulations, results=results)


def _record_lab_state_reset(run: _Run) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Reset one Lab-path environment of the delayed island and record each group's effort and PD demand.

    Reset environments must accept fresh commands while the remaining environments retain their delay history.
    """
    articulation = run.articulations["delayed"]
    commands = articulation.actuators.target_command
    old_target = commands.position.torch.clone()
    # Episode reset samples the configured lag; construction alone leaves actuator lag at zero.
    articulation.reset()
    commands.set_position_index(value=old_target)
    articulation.write_data_to_sim()
    new_target = old_target + 0.02
    articulation.reset(env_ids=torch.tensor([0], device=articulation.device))
    commands.set_position_index(value=new_target)
    articulation.write_data_to_sim()
    expected_target = old_target.clone()
    expected_target[0] = new_target[0]
    recorded = {}
    for name, actuator in articulation.actuators.items():
        joints = actuator.joint_indices
        demand = (
            actuator.stiffness * (expected_target[:, joints] - articulation.data.joint_pos.torch[:, joints])
            - actuator.damping * articulation.data.joint_vel.torch[:, joints]
        )
        recorded[name] = (actuator.computed_effort.clone(), demand)
    return recorded


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def device(request: pytest.FixtureRequest) -> str:
    """Simulation device of the actuator scenes."""
    return request.param


@pytest.fixture(scope="module")
def lab_run(device: str) -> dict:
    """Roll out every shared island on the Isaac Lab actuator path."""
    islands = {name: island for name, island in _islands().items() if not island.newton_only}
    sim_cfg = newton_sim_cfg(device, dt=DT, use_newton_actuators=False, solver_cfg=_solver_cfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        run = _run(sim, islands, use_newton_actuators=False)
        return {"results": run.results, "state_reset": _record_lab_state_reset(run)}


@pytest.fixture(scope="module")
def decimated_runs(device: str) -> dict[str, dict]:
    """Roll out the RemotizedPD island with decimation 2 and CUDA-graph capture on both execution paths."""
    islands = {"remotized": _Island(REMOTIZED_PD_ACTUATORS, ramp_targets=True)}
    runs = {}
    for use_newton_actuators in (False, True):
        sim_cfg = newton_sim_cfg(
            device,
            dt=1.0 / 100.0,
            use_newton_actuators=use_newton_actuators,
            solver_cfg=_solver_cfg(),
            num_substeps=2,
            use_cuda_graph=True,
        )
        with build_simulation_context(sim_cfg=sim_cfg) as sim:
            run = _run(sim, islands, use_newton_actuators, dt=1.0 / 100.0, num_steps=5, decimation=2)
            runs["newton" if use_newton_actuators else "lab"] = run.results["remotized"]
    return runs


@pytest.fixture(scope="module")
def newton_run(device: str, lab_run: dict, decimated_runs: dict) -> Iterator[_Run]:
    """Roll out every island on the Newton actuator path and keep the scene alive.

    Depends on the Lab-path and decimated runs so their scenes are built and closed first.
    """
    mlp_path = make_dummy_mlp_checkpoint()
    lstm_path = make_dummy_lstm_checkpoint()
    sim_cfg = newton_sim_cfg(device, dt=DT, use_newton_actuators=True, solver_cfg=_solver_cfg())
    try:
        with build_simulation_context(sim_cfg=sim_cfg) as sim:
            yield _run(sim, _islands(mlp_path, lstm_path), use_newton_actuators=True)
    finally:
        os.unlink(mlp_path)
        os.unlink(lstm_path)


def _assert_equivalent(
    lab_result: dict, newton_result: dict, torque_atol: float = EquivalenceAssertionsMixin.torque_atol
) -> None:
    """Run the shared trajectory and telemetry oracles on one island's Lab and Newton rollouts.

    Args:
        lab_result: Rollout recorded on the Isaac Lab actuator path.
        newton_result: Rollout recorded on the Newton actuator path.
        torque_atol: Absolute tolerance of the torque-telemetry oracles [N·m].
    """
    oracle = EquivalenceAssertionsMixin()
    oracle.torque_atol = torque_atol
    oracle.lab_result = lab_result
    oracle.newton_result = newton_result
    oracle.test_joint_positions_match()
    oracle.test_joint_velocities_match()
    oracle.test_applied_effort_match()
    oracle.test_computed_effort_match()


# ---------------------------------------------------------------------------
# Equivalence tests with different actuator types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "island",
    ["ideal", "cartpole", "dc_motor", "mixed", "implicit_feedforward", "delayed", "remotized"],
)
def test_newton_actuators_match_lab_actuators(lab_run: dict, newton_run: _Run, island: str) -> None:
    """Newton-native actuators reproduce the Isaac Lab actuator trajectories and torque telemetry.

    The islands cover IdealPD on the floating leg and on the fixed cartpole, which share the model-wide adapter,
    a saturating DC motor, implicit hips beside explicit Newton actuators, an implicit feedforward effort added on
    top of the solver's joint drive, and command delays with and without position-based clamping.
    """
    _assert_equivalent(
        lab_run["results"][island], newton_run.results[island], torque_atol=_islands()[island].torque_atol
    )


def test_decimated_remotized_pd_matches_lab_actuators(decimated_runs: dict) -> None:
    """RemotizedPD with decimation 2 and CUDA-graph capture: Lab vs Newton."""
    _assert_equivalent(decimated_runs["lab"], decimated_runs["newton"])


def test_dc_motor_clamp_binds(lab_run: dict, newton_run: _Run) -> None:
    """The saturating DC motor clamps on both paths, so the equivalence can detect missing clamping."""
    for result in (lab_run["results"]["dc_motor"], newton_run.results["dc_motor"]):
        assert any(
            not torch.allclose(applied, computed)
            for applied, computed in zip(result["applied_effort"], result["computed_effort"])
        ), "the DC-motor clamp never bound, so the equivalence cannot detect missing clamping"


@pytest.mark.parametrize(
    "island, group_names, controller_type, clamping, has_delay",
    [
        ("delayed", ("legs",), "DrivePD", None, True),
        ("remotized", ("knees",), "DrivePD", "ClampingPositionBased", True),
        ("neural", ("mlp_legs",), "DriveNeuralMLP", "ClampingDCMotor", False),
        ("neural", ("lstm_legs",), "DriveNeuralLSTM", "ClampingDCMotor", False),
    ],
    ids=["delayed_pd", "remotized_pd", "mlp", "lstm"],
)
def test_newton_actuator_authoring(
    newton_run: _Run, island: str, group_names: tuple[str, ...], controller_type: str, clamping: str | None, has_delay
) -> None:
    """Lab actuator configurations are authored as the matching Newton controller, clamping, and delay."""
    info = [entry for entry in newton_run.results[island]["actuator_info"] if entry["group"] in group_names]
    assert info, f"no Newton actuators were created for {group_names}"
    for entry in info:
        assert entry["controller_type"] == controller_type
        assert clamping is None or clamping in entry["clamping_types"]
        assert entry["has_delay"] is has_delay
    if island == "neural":
        # the neural controllers run without producing non-finite joint positions
        assert all(torch.isfinite(pos).all() for pos in newton_run.results[island]["joint_pos"])


# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------


def test_newton_actuator_rollout_matches_reordered_joints(newton_run: _Run) -> None:
    """Match Newton-backend actuator traces under a permuted public joint ordering.

    The implicit hip group reads its torque telemetry from backend-order effort buffers, so the
    permutation-sensitive feedforward effort also checks that gather.
    """
    identity_result = newton_run.results["mixed_permuted"]
    reordered_result = newton_run.results["mixed_permuted_reordered"]

    installed_ordering = reordered_result["joint_ordering"]
    assert installed_ordering is not None
    assert not installed_ordering["is_identity"]
    assert_articulation_ordering_trace_matches(identity_result, reordered_result, _LEG_REORDERED_JOINT_NAMES)


@pytest.mark.parametrize("island", ["ideal", "ideal_reordered"], ids=["identity", "reordered"])
def test_write_data_to_sim_writes_joint_targets_in_backend_order(newton_run: _Run, island: str) -> None:
    """Explicit actuators on the Newton-actuator path publish their raw targets to the backend in backend order."""
    articulation = newton_run.articulations[island]
    assert (articulation.data.joint_ordering is not None) is (island == "ideal_reordered")
    assert articulation._has_newton_actuators is True

    # Distinct per-joint targets away from the defaults, so a skipped or unpermuted write is visible.
    target = articulation.data.default_joint_pos.torch.clone()
    target += 0.01 * torch.arange(1, articulation.num_joints + 1, device=target.device)
    articulation.set_joint_position_target_index(target=target)
    articulation.write_data_to_sim()

    source = articulation.actuators.target_command.position.torch
    torch.testing.assert_close(source, target)
    user_to_backend = (
        list(articulation.joint_ordering.user_to_backend_indices)
        if articulation.joint_ordering is not None
        else list(range(articulation.num_joints))
    )
    expected_backend_target = torch.empty_like(source)
    expected_backend_target[:, user_to_backend] = source
    torch.testing.assert_close(wp.to_torch(articulation.data._sim_bind_joint_position_target), expected_backend_target)


def test_newton_native_actuator_gain_write_maps_public_joint_subset_to_backend(newton_run: _Run) -> None:
    """Map selected public joint IDs to Newton-controller columns."""
    articulation = newton_run.articulations["ideal_reordered"]
    assert articulation.joint_ordering is not None
    assert articulation.newton_actuator_adapter is not None

    def gather_stiffness() -> torch.Tensor:
        stiffness = torch.zeros((articulation.num_instances, articulation.num_joints), device=articulation.device)
        for actuator in articulation.newton_actuator_adapter.actuators:
            if hasattr(actuator.controller, "kp"):
                stiffness += wp.to_torch(
                    articulation.root_view.get_actuator_parameter(actuator, actuator.controller, "kp")
                )
        return stiffness

    stiffness_before = gather_stiffness()
    env_ids = torch.tensor([1], device=articulation.device, dtype=torch.long)
    joint_ids = torch.tensor([1, 3, 4], device=articulation.device, dtype=torch.long)
    stiffness = torch.tensor([[101.0, 103.0, 104.0]], device=articulation.device)

    with pytest.warns(DeprecationWarning, match="write_actuator_stiffness_to_sim"):
        articulation.write_actuator_stiffness_to_sim(stiffness=stiffness, env_ids=env_ids, joint_ids=joint_ids)

    backend_joint_ids = torch.tensor(
        articulation.joint_ordering.user_to_backend_indices, device=articulation.device, dtype=torch.long
    )[joint_ids]
    expected_stiffness = stiffness_before.clone()
    expected_stiffness[env_ids.unsqueeze(1), backend_joint_ids.unsqueeze(0)] = stiffness
    torch.testing.assert_close(gather_stiffness(), expected_stiffness)


# ---------------------------------------------------------------------------
# Domain randomization via events.py — Newton backend
# ---------------------------------------------------------------------------


def test_randomize_actuator_gains_reaches_newton_controllers(newton_run: _Run) -> None:
    """``randomize_actuator_gains`` writes kp/kd into the selected environment of one articulation's controllers.

    The event writes through ``write_group_parameter`` and the assertions read back through the public
    ``read_group_parameter``. A degenerate ``(K, K)`` range with ``operation="abs"`` sets each randomized cell to
    exactly ``K``.
    """
    groups = {
        "legs": (newton_run.articulations["ideal"], "legs"),
        "cartpole": (newton_run.articulations["cartpole"], "all_joints"),
    }
    legs = groups["legs"][0]
    assert SimulationManager._adapter is not None

    def gains(name: str) -> torch.Tensor:
        """Return the ``(kp, kd)`` gains of one articulation's actuator group, shape ``(2, num_envs, num_joints)``."""
        articulation, group = groups[name]
        return torch.stack([read_group_parameter(articulation.actuators, group, "controller", p) for p in ("kp", "kd")])

    # Every environment reads the configured gains. On the floating leg this pins the env-major DOF stride
    # decoding (6 free-root DOFs + leg joints): a wrong stride corrupts every environment past the first.
    configured = torch.tensor([40.0, 5.0], device=legs.device).view(2, 1, 1).expand(2, NUM_ENVS, legs.num_joints)
    torch.testing.assert_close(gains("legs"), configured)

    env = MockEnv({name: articulation for name, (articulation, _) in groups.items()}, NUM_ENVS, legs.device)
    for name in ("cartpole", "legs"):
        expected = {other: gains(other).clone() for other in groups}
        expected[name][:, 0] = torch.tensor([100.0, 7.0], device=legs.device).view(2, 1)
        term, asset_cfg = build_dr_term(env, name)
        term(
            env,
            env_ids=torch.tensor([0], device=legs.device, dtype=torch.long),
            asset_cfg=asset_cfg,
            stiffness_distribution_params=(100.0, 100.0),
            damping_distribution_params=(7.0, 7.0),
            operation="abs",
            distribution="uniform",
        )
        # only environment 0 of the selected articulation changes
        for other in groups:
            torch.testing.assert_close(gains(other), expected[other])


# ---------------------------------------------------------------------------
# Per-env reset: actuator state isolation
# ---------------------------------------------------------------------------


def test_newton_state_reset_isolated_to_reset_env(newton_run: _Run) -> None:
    """Newton: ``num_pushes`` zeroes for env 0's DOFs only after reset of [0].

    Only the reset articulation's own delayed actuators are checked. The adapter is model-wide, so the other
    islands' actuators share its state buffers; their state is not part of this articulation's contract.
    """
    articulation = newton_run.articulations["delayed"]
    adapter = SimulationManager._adapter
    assert adapter is not None
    own_actuators = []
    for group_name in articulation.actuators._native_group_names:
        group_actuators = articulation.actuators[group_name]
        own_actuators.extend(group_actuators if isinstance(group_actuators, tuple) else (group_actuators,))
    stateful_pairs = [
        (act, st)
        for act, st in zip(adapter.actuators, adapter._states_a)
        if any(act is own for own in own_actuators) and st is not None and st.delay_state is not None
    ]
    assert len(stateful_pairs) > 0, "expected at least one DelayedPD actuator with delay_state"

    for act, state in stateful_pairs:
        pushes_before = state.delay_state.num_pushes.numpy()
        assert (pushes_before > 0).all(), "expected non-zero num_pushes for all DOFs after warmup"

    articulation.reset(env_ids=torch.tensor([0], device=articulation.device, dtype=torch.long))

    # Map each entry of ``act.indices`` to its env via the adapter's per-env DOF count. The adapter is
    # model-wide (includes free-joint DOFs on floating-base articulations), so ``adapter.num_joints`` is the
    # stride.
    for act, state in stateful_pairs:
        pushes_after = state.delay_state.num_pushes.numpy()
        indices_np = act.indices.numpy()
        for i, global_dof in enumerate(indices_np):
            env = int(global_dof) // adapter.num_joints
            if env == 0:
                assert int(pushes_after[i]) == 0, f"DOF {i} (env {env}) should be reset to 0, got {pushes_after[i]}"
            else:
                assert int(pushes_after[i]) > 0, f"DOF {i} (env {env}) was NOT in reset env_ids but num_pushes is 0"


def test_lab_state_reset_isolated_to_reset_env(lab_run: dict) -> None:
    """Reset environments accept fresh commands while the remaining environments retain their history."""
    for computed_effort, demand in lab_run["state_reset"].values():
        torch.testing.assert_close(computed_effort, demand)
