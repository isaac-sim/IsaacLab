# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end checks of Newton-native BAM on live MJWarp articulations."""

from dataclasses import fields

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonManager, NewtonSolverCfg
from isaaclab_newton.physics.mjwarp_actuator_bridge import MjWarpActuatorBridge
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo
from newton_test_utils import local_usd, spawn_assets

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.actuators.newton import read_group_parameter, write_group_parameter
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kitless,
    pytest.mark.filterwarnings("error:.*deprecated in Newton 1[.]6:DeprecationWarning"),
]

DT = 1.0 / 120.0
"""Physics timestep [s]."""

BAM_DATA_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Tests/BAM"
"""Hosted BAM fixtures: the pendulum USD, its recorded trajectory and the upstream fit."""

NUM_ENVS = 2
"""Environments simulated side by side, so a per-environment randomization is observable."""

VIN = 7.4
"""Supply voltage the actuator is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the actuator is configured with [-]."""

INITIAL_ANGLE = 0.3
"""Angle the arm is released from [rad]. The commanded target is always 0."""


def _make_sim_cfg(
    device: str, solver_cfg: NewtonSolverCfg | None = None, use_newton_actuators: bool = True
) -> SimulationCfg:
    """Build the simulation configuration, with this module's MJWarp solver unless *solver_cfg* is given."""
    if solver_cfg is None:
        solver_cfg = MJWarpSolverCfg(njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1)
    return SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=2, debug_mode=False),
    )


@pytest.fixture(scope="module")
def pendulum_usd() -> str:
    """Local path of the fixed-base BAM pendulum fixture."""
    return retrieve_file_path(f"{BAM_DATA_DIR}/bam_pendulum.usda")


@pytest.fixture
def native_sim(device):
    """Newton simulation context running the Newton-native actuator path."""
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=False, sim_cfg=_make_sim_cfg(device)
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


def _make_cfg(**overrides) -> BamActuatorCfg:
    """Use the identified fit recorded alongside the upstream reference outputs."""
    with np.load(retrieve_file_path(f"{BAM_DATA_DIR}/bam_xl330_m6_goldens.npz")) as data:
        params = {key.removeprefix("attr_"): data[key].item() for key in data.files if key.startswith("attr_")}
    params["resistance"] = params.pop("R")
    motor = BamMotorCfg(model="m6", **{f.name: params[f.name] for f in fields(BamMotorCfg) if f.name in params})
    kwargs = {"joint_names_expr": [".*"], "motor": motor, "vin": VIN, "kp_fw": KP_FW}
    kwargs.update(overrides)
    return BamActuatorCfg(**kwargs)


def _spawn_pendulums(sim, pendulum_usd: str, cfgs: dict[str, BamActuatorCfg]) -> list[Articulation]:
    """Spawn one BAM pendulum per entry of *cfgs* in each of :data:`NUM_ENVS` environments and reset."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robots = [
        Articulation(
            ArticulationCfg(
                prim_path=f"/World/Env_[^/]*/{name}",
                spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                actuators={"servo": cfg},
            )
        )
        for name, cfg in cfgs.items()
    ]
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg for robot in robots], NUM_ENVS, 1.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    return robots


def _spawn_branching(sim, actuators: dict[str, BamActuatorCfg]) -> Articulation:
    """Spawn the four-joint branching fixture in PhysX joint order, which interleaves its arms' backend joints."""
    cfg = ArticulationCfg(
        prim_path="/World/Env_.*/Robot",
        spawn=local_usd("articulation_ordering_branching.usda"),
        joint_ordering="physx",
        actuators=actuators,
    )
    robot = spawn_assets({"robot": cfg})["robot"]
    sim.reset()
    return robot


@pytest.mark.parametrize(
    "device,decimation",
    [
        (device, decimation)
        for device in test_devices()
        for decimation in ((1, 2) if device.startswith("cuda") else (1,))
    ],
)
def test_native_pendulum_matches_recorded_trajectory(native_sim, device, pendulum_usd, decimation):
    """Replay command reversals, settling and per-environment friction changes.

    On CUDA the friction scales change after graph capture, so live parameter updates are
    exercised with both state-buffer parities.
    """
    (robot,) = _spawn_pendulums(native_sim, pendulum_usd, {"Robot": _make_cfg()})
    drive = robot.actuators["servo"].drive
    np.testing.assert_allclose(
        robot.data.joint_viscous_friction_coeff.torch.cpu().numpy(), _make_cfg().motor.friction_viscous, rtol=1e-6
    )
    NewtonManager.set_decimation(decimation)
    robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, INITIAL_ANGLE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
    robot.actuators.reset()
    with np.load(retrieve_file_path(f"{BAM_DATA_DIR}/bam_pendulum_trajectory.npz")) as golden:
        assert native_sim.get_physics_dt() == float(golden["dt"])
        traces = {name: [] for name in ("position", "velocity", "effort", "friction_budget")}
        for index in range(0, len(golden["target"]), decimation):
            if index == 16:
                # A joint-property resync must preserve the initialized passive damping.
                robot.write_joint_armature_to_sim_index(armature=robot.data.joint_armature.torch.clone())
            target = float(golden["target"][index])
            robot.actuators.target_command.set_position_index(value=torch.full_like(robot.data.joint_pos.torch, target))
            write_group_parameter(
                robot.actuators,
                "servo",
                "drive",
                "friction_scale",
                torch.as_tensor(golden["friction_scale"][index, :, None], device=robot.device),
            )
            robot.write_data_to_sim()
            native_sim.step()
            robot.update(native_sim.get_physics_dt() * decimation)
            traces["position"].append(robot.data.joint_pos.torch.cpu().numpy().copy())
            traces["velocity"].append(robot.data.joint_vel.torch.cpu().numpy().copy())
            traces["effort"].append(robot.actuators.applied_effort.torch.cpu().numpy().copy())
            traces["friction_budget"].append(drive.friction_budget.numpy().copy())
        for name, values in traces.items():
            # CPU and CUDA solver reductions differ slightly.
            np.testing.assert_allclose(
                np.stack(values), golden[name][decimation - 1 :: decimation], atol=2e-5, rtol=2e-4, err_msg=name
            )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_robots_pair_with_their_own_actuators_across_hard_resets(native_sim, device, pendulum_usd):
    """Identical robots share one bound actuator, and a differing robot samples its own start-up ranges.

    The samples hold across environment resets; hard resets rebind without stacking solver hooks.
    """
    sampled_cfg = _make_cfg(max_delay=2, vin_range=(6.0, 8.0), vin_drop_gain_range=(0.0, 0.2))
    cfgs = {"RobotA": _make_cfg(), "RobotB": _make_cfg(), "RobotC": sampled_cfg}
    robot_a, robot_b, robot_c = _spawn_pendulums(native_sim, pendulum_usd, cfgs)
    assert robot_c.actuators["servo"] is not robot_a.actuators["servo"], "differing max_delay must not merge"
    voltage = read_group_parameter(robot_a.actuators, "servo", "drive", "vin")
    torch.testing.assert_close(voltage, torch.full_like(voltage, VIN), atol=1e-6, rtol=0.0)
    sampled = {}
    for attr, (low, high) in (("vin", sampled_cfg.vin_range), ("sag_gain", sampled_cfg.vin_drop_gain_range)):
        values = read_group_parameter(robot_c.actuators, "servo", "drive", attr)
        assert bool(((values >= low) & (values <= high)).all()), f"{attr} outside its configured range"
        assert len(torch.unique(values)) > 1, f"{attr} drew the same value for every environment"
        sampled[attr] = values.clone()
    robot_c.actuators.reset()
    for attr, values in sampled.items():
        torch.testing.assert_close(read_group_parameter(robot_c.actuators, "servo", "drive", attr), values)
    torch.testing.assert_close(
        read_group_parameter(robot_c.actuators, "servo", "drive", "friction_scale"),
        torch.ones(NUM_ENVS, robot_c.num_joints, device=robot_c.device),
    )

    hook_counts = (len(NewtonManager._pre_actuator_callbacks), len(NewtonManager._post_actuator_callbacks))
    for reset_index in range(3):
        if reset_index:
            # A hard reset rebuilds the model and drives; the bridge hooks must be replaced, not appended.
            native_sim.reset()
            assert hook_counts == (
                len(NewtonManager._pre_actuator_callbacks),
                len(NewtonManager._post_actuator_callbacks),
            )
        assert robot_a.actuators["servo"] is robot_b.actuators["servo"], "identical robots must merge"
        for robot in (robot_a, robot_c):
            assert robot.actuators["servo"].drive.external_torque is not None
        native_sim.step()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_each_group_configures_only_its_own_actuator(native_sim, device):
    """BAM groups of one robot in separate Newton actuators each get their own start-up configuration."""
    elbows = _make_cfg(joint_names_expr=[".*_elbow"], max_delay=2, vin_range=(6.5, 6.5), stiff_frictionloss=False)
    robot = _spawn_branching(native_sim, {"shoulders": _make_cfg(joint_names_expr=[".*_shoulder"]), "elbows": elbows})
    assert robot.actuators["shoulders"] is not robot.actuators["elbows"], "differing max_delay must not merge"

    # The shoulders keep the nominal supply and the stiff friction constraint; the elbows use their own settings.
    for name, expected in (("shoulders", VIN), ("elbows", 6.5)):
        voltage = read_group_parameter(robot.actuators, name, "drive", "vin")
        torch.testing.assert_close(voltage, torch.full_like(voltage, expected), atol=1e-6, rtol=0.0)
    solref = NewtonManager.backend.model.mujoco.solreffriction.numpy()
    stiff = np.asarray(MjWarpActuatorBridge.STIFF_SOLREF_FRICTION)
    assert np.allclose(solref[robot.actuators["shoulders"].indices.numpy()], stiff)
    assert not np.isclose(solref[robot.actuators["elbows"].indices.numpy()], stiff).all(axis=1).any()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_groups_sharing_one_actuator_reject_conflicting_settings(native_sim, device):
    """Groups Newton merges into one drive cannot carry different start-up ranges."""
    cfgs = {
        "shoulders": _make_cfg(joint_names_expr=[".*_shoulder"]),
        "elbows": _make_cfg(joint_names_expr=[".*_elbow"], vin_range=(6.5, 6.5)),
    }
    with pytest.raises(ValueError, match="share a Newton actuator"):
        _spawn_branching(native_sim, cfgs)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize(
    "solver_cfg, use_newton_actuators, overrides, error",
    [
        (FeatherstoneSolverCfg(), True, [{}], "BAM actuators require.*MJWarp"),
        (None, False, [{}], "use_newton_actuators"),
        (
            None,
            True,
            [{"backlash_joint_template": "passive_{}_backlash"}],
            "BAM backlash requires.*passive_joint_backlash",
        ),
        (None, True, [{}, {"vin_drop_gain_range": (0.0, 0.2)}], "sharing a Newton BAM actuator"),
    ],
    ids=["featherstone", "lab-executed", "missing-play-hinge", "conflicting-robots"],
)
def test_unsupported_setups_fail_at_startup(device, pendulum_usd, solver_cfg, use_newton_actuators, overrides, error):
    """Setups BAM cannot run fail while the scene starts, instead of approximating friction or dropping settings."""
    sim_cfg = _make_sim_cfg(device, solver_cfg=solver_cfg, use_newton_actuators=use_newton_actuators)
    with build_simulation_context(device=device, gravity_enabled=True, add_ground_plane=False, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None  # noqa: SLF001
        cfgs = {f"Robot{index}": _make_cfg(**kwargs) for index, kwargs in enumerate(overrides)}
        with pytest.raises(ValueError, match=error):
            _spawn_pendulums(sim, pendulum_usd, cfgs)


def test_bridge_rejects_cpu_mujoco():
    """CPU MuJoCo builds the MuJoCo Warp model too, but steps the host model the bridge never reaches."""
    builder = ModelBuilder()
    body = builder.add_link(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_revolute(-1, body, axis=(0.0, 0.0, 1.0))])
    solver = SolverMuJoCo(builder.finalize(device="cpu"), use_mujoco_cpu=True)
    assert solver.mjw_model is not None
    assert not MjWarpActuatorBridge.is_available(solver)
