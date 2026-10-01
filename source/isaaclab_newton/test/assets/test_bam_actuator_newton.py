# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end checks of Newton-native BAM on live MJWarp articulations.

Recorded pendulum trajectories cover motor response, settling and live friction changes.
No Isaac Sim runtime or downloaded asset is needed.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonManager,
)

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg
from isaaclab.actuators.newton import DriveBam, read_group_parameter, write_group_parameter
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kitless,
    pytest.mark.filterwarnings("error:.*deprecated in Newton 1[.]6:DeprecationWarning"),
]

PENDULUM_USDA = """\
#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Pivot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.001, 0.001, 0.001)
    }

    def Xform "Arm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.02
        point3f physics:centerOfMass = (0.05, 0, 0)
        float3 physics:diagonalInertia = (0.002, 0.002, 0.002)
    }

    def PhysicsFixedJoint "anchor"
    {
        rel physics:body1 = </Robot/Pivot>
    }

    def PhysicsRevoluteJoint "joint"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Pivot>
        rel physics:body1 = </Robot/Arm>
    }
}
"""
"""Fixed-base single-degree-of-freedom pendulum.

``Pivot`` is welded to the world by ``anchor``, a fixed joint whose ``physics:body0`` is left
unset. That weld is what makes the USD physics parser report an articulation at all: a lone
body hanging off a world-anchored revolute joint is imported as an orphan joint and no
articulation is created, so :class:`~isaaclab.assets.Articulation` finds nothing to bind to.
``Arm`` is the only degree of freedom.

``Arm``'s centre of mass is offset along +X and the joint spins about +Y, so rotating the
joint by ``theta`` swings the centre of mass down and gravity applies a pure
``m * g * L * cos(theta)`` load about the joint axis.

``Arm``'s inertia is authored independently of its mass distribution and is deliberately much
larger than ``m * L^2``: it stands in for the rotor inertia a geared servo reflects to its
output shaft (0.0018 kg m^2 for the Dynamixel XL330 the BAM parameters were fitted on).
Without it the joint inertia would sit far below the actuator's electrical damping times the
timestep, and a back-EMF torque applied explicitly by the actuator could not be integrated
stably.
"""

DT = 1.0 / 120.0
"""Physics timestep [s]."""

NUM_ENVS = 2
"""Environments simulated side by side, so a per-environment randomization is observable."""

NUM_STEPS = 200
"""Steps each settling phase runs for [-]. The joint is at rest well before this."""

VIN = 7.4
"""Supply voltage the actuator is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the actuator is configured with [-]."""

INITIAL_ANGLE = 0.3
"""Angle the arm is released from [rad]. The commanded target is always 0."""


def _make_sim_cfg(device: str, use_newton_actuators: bool = False) -> SimulationCfg:
    """Build the MJWarp configuration used by every test in this module."""
    return SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1),
            num_substeps=2,
            debug_mode=False,
        ),
    )


@pytest.fixture(scope="module")
def pendulum_usd(tmp_path_factory) -> str:
    """Write :data:`PENDULUM_USDA` to a temporary file and return its path."""
    path = tmp_path_factory.mktemp("bam_pendulum") / "single_joint_pendulum.usda"
    path.write_text(PENDULUM_USDA)
    stage = Usd.Stage.Open(str(path))
    prim = stage.DefinePrim("/Robot/asset_actuator", "NewtonActuator")
    fixture = Path(__file__).resolve().parents[3] / "isaaclab/test/actuators/data/bam_xl330_m6.usda"
    prim.GetReferences().AddReference(str(fixture), "/BamActuator")
    prim.CreateRelationship("newton:targets").SetTargets(["/Robot/joint"])
    stage.GetRootLayer().Save()
    return str(path)


@pytest.fixture
def sim(device):
    """Newton simulation context running the Isaac Lab-executed actuator path."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


@pytest.fixture
def native_sim(device):
    """Newton simulation context running the Newton-native actuator path."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device, use_newton_actuators=True),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


def _release(robot: Articulation) -> None:
    """Put every arm back at :data:`INITIAL_ANGLE` at rest and clear the actuator state."""
    robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, INITIAL_ANGLE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
    robot.actuators.reset()


def _build_native_pendulum(sim, pendulum_usd: str, actuator_cfg: BamActuatorCfg | None = None) -> Articulation:
    """Spawn :data:`NUM_ENVS` BAM-driven pendulums on the Newton-native actuator path.

    Args:
        sim: Simulation context to spawn into.
        pendulum_usd: Path of the pendulum asset.
        actuator_cfg: Servo group to drive them with. Defaults to the plain BAM configuration.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={"servo": actuator_cfg or BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)},
        )
    )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg], NUM_ENVS, 1.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert robot.is_initialized
    assert "servo" in robot.actuators._native_group_names, "the BAM group must run on the Newton path"
    return robot


def _native_drive(robot: Articulation) -> DriveBam:
    """Return the BAM drive the articulation's native group is executed by."""
    drives = [actuator.drive for actuator in NewtonManager._adapter.actuators if isinstance(actuator.drive, DriveBam)]
    assert len(drives) == 1, "the fixture has exactly one BAM actuator"
    return drives[0]


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

    The first 32 steps were recorded at 3691a0bf04 and remain unchanged. The extension
    was recorded eagerly at db06f2bd9b and checked against the original static-friction
    bounds before removing those helpers. It adds 200 settling steps followed by 200
    steps with distinct friction scales. CUDA replays change those scales after capture,
    exercising live parameter updates with both odd and even state-buffer parity.
    """
    robot = _build_native_pendulum(native_sim, pendulum_usd)
    drive = _native_drive(robot)
    NewtonManager.set_decimation(decimation)
    _release(robot)
    with np.load(Path(__file__).parent / "data" / "bam_pendulum_trajectory.npz") as golden:
        assert native_sim.get_physics_dt() == float(golden["dt"])
        traces = {name: [] for name in ("position", "velocity", "effort", "friction_budget")}
        for index in range(0, len(golden["target"]), decimation):
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
    if device.startswith("cuda"):
        assert NewtonManager._graph is not None, "the trajectory must exercise graph replay"


def _build_two_native_pendulums(sim, pendulum_usd: str, second_cfg: BamActuatorCfg) -> tuple:
    """Spawn two BAM articulations per environment and initialize the simulation.

    Both robots are the same asset, so Newton merges their BAM joints into *one* actuator
    whenever the configurations agree on every grouping-key field -- which is exactly the
    multi-robot case the per-articulation binding has to get right.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robots = []
    for name, cfg in (
        ("RobotA", BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)),
        ("RobotB", second_cfg),
    ):
        robots.append(
            Articulation(
                ArticulationCfg(
                    prim_path=f"/World/Env_[^/]*/{name}",
                    spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                    actuators={"servo": cfg},
                )
            )
        )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg for robot in robots], NUM_ENVS, 1.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    return tuple(robots)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_two_articulations_sharing_one_actuator_must_agree(native_sim, device, pendulum_usd):
    """Two robots merged into one Newton actuator cannot carry different BAM settings.

    ``vin_range``, ``vin_drop_gain_range``, ``friction_scale_range`` and ``stiff_frictionloss``
    are not part of Newton's actuator-grouping key, so structurally identical robots share one
    actuator and one set of parameter arrays. Applying the second articulation's ranges would
    silently discard the first's randomization, and skipping them would silently ignore the
    second's configuration, so the conflict has to be refused.
    """
    conflicting = BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW, friction_scale_range=(0.5, 2.0))
    with pytest.raises(ValueError, match="share one Newton actuator"):
        _build_two_native_pendulums(native_sim, pendulum_usd, conflicting)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_two_articulations_with_matching_settings_bind_once(native_sim, device, pendulum_usd):
    """Agreeing robots share the actuator, and neither is left unbound or bound twice."""
    matching = BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)
    robot_a, robot_b = _build_two_native_pendulums(native_sim, pendulum_usd, matching)

    bam_actuators = [actuator for actuator in NewtonManager._adapter.actuators if isinstance(actuator.drive, DriveBam)]
    assert len(bam_actuators) == 1, "the identical robots must merge into one Newton actuator"
    drive = bam_actuators[0].drive
    assert drive.external_torque is not None

    # One binding, not one per articulation: a second registration would write the same budget
    # twice per step and, worse, hide a scoping mistake. The gather hook is the countable half
    # of the pair -- the publish hook is registered as a lambda, so it cannot be told apart from
    # any other post-actuator callback -- and both are registered together or not at all.
    assert len(NewtonManager._pre_actuator_callbacks) == 1

    for robot in (robot_a, robot_b):
        assert "servo" in robot.actuators._native_group_names


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_startup_ranges_are_sampled_per_environment(native_sim, device, pendulum_usd):
    """Apply startup ranges per environment and preserve an unset parameter's nominal value."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={
                "servo": BamActuatorCfg(
                    joint_names_expr=[".*"],
                    kp_fw=KP_FW,
                    vin_range=(6.0, 8.0),
                    friction_scale_range=(0.5, 1.5),
                )
            },
        )
    )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg], NUM_ENVS, 1.0)
    replicate(native_sim.get_clone_plan())
    native_sim.reset()
    assert robot.is_initialized

    for attr, (low, high) in (("vin", (6.0, 8.0)), ("friction_scale", (0.5, 1.5))):
        values = read_group_parameter(robot.actuators, "servo", "drive", attr)
        assert bool(((values >= low) & (values <= high)).all()), f"{attr} outside its configured range"
        assert len(torch.unique(values)) > 1, f"{attr} drew the same value for every environment"
    # An unset range keeps the authored nominal.
    torch.testing.assert_close(
        read_group_parameter(robot.actuators, "servo", "drive", "sag_gain"),
        torch.zeros(NUM_ENVS, robot.num_joints, device=robot.device),
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_each_articulation_configures_only_its_own_actuator(native_sim, device, pendulum_usd):
    """Two robots that do *not* merge must each get their own configuration.

    The actuator adapter is simulation-global, so an articulation that walked the whole
    adapter would apply its own start-up ranges to another robot's actuator -- silently, and
    with whichever articulation initialized first winning. Differing ``max_delay`` puts the two
    robots in different Newton actuators; only the scoping decides which one each configures.
    """
    delayed = BamActuatorCfg(
        joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW, max_delay=2, friction_scale_range=(3.0, 3.0)
    )
    robot_a, robot_b = _build_two_native_pendulums(native_sim, pendulum_usd, delayed)
    assert len({id(actuator) for actuator in NewtonManager._adapter.actuators}) == 2, (
        "the two robots must not merge, or the test cannot tell the configurations apart"
    )

    # robot_a keeps ``_build_two_native_pendulums``'s default cfg (no start-up range, so 1.0).
    for robot, expected in ((robot_a, 1.0), (robot_b, 3.0)):
        friction_scale = read_group_parameter(robot.actuators, "servo", "drive", "friction_scale")
        torch.testing.assert_close(friction_scale, torch.full_like(friction_scale, expected), atol=1e-6, rtol=0.0)


def test_bam_rejects_a_non_mjwarp_solver(pendulum_usd):
    """Reject BAM during solver initialization instead of approximating solver friction."""
    device = "cpu"
    sim_cfg = SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=True,
        physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), num_substeps=2, debug_mode=False),
    )
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=False, sim_cfg=sim_cfg
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        for index in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Env_[^/]*/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                actuators={"servo": BamActuatorCfg(joint_names_expr=[".*"], kp_fw=KP_FW, vin_range=(6.0, 8.0))},
            )
        )
        clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg], NUM_ENVS, 1.0)
        replicate(sim_ctx.get_clone_plan())
        with pytest.raises(ValueError, match="BAM actuators require.*MJWarp"):
            sim_ctx.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_bam_cfg_is_refused_on_the_isaac_lab_executed_path(sim, device, pendulum_usd):
    """BAM requires the native actuator loop."""
    with pytest.raises(ValueError, match="use_newton_actuators"):
        _build_native_pendulum(sim, pendulum_usd, BamActuatorCfg(joint_names_expr=[".*"], kp_fw=KP_FW))
