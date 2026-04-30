# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensor, JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.ant import ANT_CFG


def _make_single_joint_articulation_cfg() -> ArticulationCfg:
    """Single-joint revolute test articulation (root ``CenterPivot`` + arm ``Arm``)."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=2000.0,
                damping=100.0,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _make_cartpole_articulation_cfg(pole_damping: float = 0.0) -> ArticulationCfg:
    """Two-joint cartpole articulation (cart + pole).

    Args:
        pole_damping: Damping for the cart-to-pole revolute joint.
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
        ),
        actuators={
            "cart_actuator": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"], effort_limit_sim=400.0, stiffness=0.0, damping=10.0
            ),
            "pole_actuator": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=pole_damping
            ),
        },
    )


@configclass
class _SingleJointSceneCfg(InteractiveSceneCfg):
    """Scene with a single-joint articulation and the joint-wrench sensor."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_single_joint_articulation_cfg()
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _CartpoleSceneCfg(InteractiveSceneCfg):
    """Scene with a cartpole (2-joint) articulation and the joint-wrench sensor."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_cartpole_articulation_cfg()
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _CartpoleDampedSceneCfg(InteractiveSceneCfg):
    """Cartpole with pole damping for steady-state physics validation tests."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_cartpole_articulation_cfg(pole_damping=10.0)
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _NestedRootAntSceneCfg(InteractiveSceneCfg):
    """Ant USD asset whose articulation root is nested under the configured asset prim."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@pytest.fixture
def sim():
    """Simulation context using the PhysX backend."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        physics=PhysxCfg(),
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None
        yield sim_ctx


def _physx_incoming_joint_wrench(sensor: JointWrenchSensor) -> torch.Tensor:
    """Read the raw PhysX incoming joint wrench tensor.

    PhysX reports spatial vectors as force followed by torque. Shape is
    ``(num_envs, num_bodies, 6)``.
    """
    raw_wrench = sensor._root_view.get_link_incoming_joint_force().view(wp.spatial_vectorf)
    return wp.to_torch(raw_wrench)


def _assert_sensor_matches_physx_tensor(sensor: JointWrenchSensor) -> None:
    """Compare sensor buffers to the raw PhysX tensor transformed into joint frames."""
    raw_wrench = _physx_incoming_joint_wrench(sensor)
    sensor_data = sensor.data

    expected_force, expected_torque = _physx_incoming_joint_wrench_in_joint_frame(sensor, raw_wrench)
    torch.testing.assert_close(sensor_data.force.torch, expected_force)
    torch.testing.assert_close(sensor_data.torque.torch, expected_torque)


def _physx_incoming_joint_wrench_in_joint_frame(
    sensor: JointWrenchSensor, raw_wrench: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw PhysX body-frame incoming joint wrenches into the configured convention."""
    force_b = raw_wrench[..., :3]
    torque_b = raw_wrench[..., 3:]
    joint_pos_b = wp.to_torch(sensor._joint_pos_b).unsqueeze(0)
    joint_quat_b = wp.to_torch(sensor._joint_quat_b).unsqueeze(0)
    torque_joint_anchor_b = torque_b - torch.cross(joint_pos_b.expand_as(force_b), force_b, dim=-1)

    flat_joint_quat_b = joint_quat_b.expand_as(raw_wrench[..., :4]).reshape(-1, 4)
    expected_force = math_utils.quat_apply_inverse(flat_joint_quat_b, force_b.reshape(-1, 3)).reshape(force_b.shape)
    expected_torque = math_utils.quat_apply_inverse(flat_joint_quat_b, torque_joint_anchor_b.reshape(-1, 3)).reshape(
        torque_b.shape
    )
    return expected_force, expected_torque


def _set_child_joint_frame(scene: InteractiveScene, child_body_name: str) -> None:
    """Set a non-identity child-side joint frame for the requested body in env 0."""
    for prim in scene.stage.Traverse():
        if not prim.GetPath().pathString.startswith("/World/envs/env_0/Robot"):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint and any(target.name == child_body_name for target in joint.GetBody1Rel().GetTargets()):
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.25, -0.15, 0.1))
            joint.GetLocalRot1Attr().Set(
                Gf.Quatf(
                    math.cos(math.pi / 4.0),
                    Gf.Vec3f(math.sin(math.pi / 4.0), 0.0, 0.0),
                )
            )
            return
    raise RuntimeError(f"Failed to find a USD joint with child body '{child_body_name}'.")


# ---------------------------------------------------------------------------
# Sensor data — pre-init contract
# ---------------------------------------------------------------------------


def test_data_before_init_is_none():
    """``force``/``torque`` return ``None`` before :meth:`create_buffers` runs."""
    from isaaclab_physx.sensors.joint_wrench import JointWrenchSensorData

    data = JointWrenchSensorData()
    assert data.force is None
    assert data.torque is None


# ---------------------------------------------------------------------------
# Initialization and shapes
# ---------------------------------------------------------------------------


def test_initialization_and_shapes(sim):
    """Sensor initializes on sim reset and exposes correctly-shaped buffers."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    # PhysX reports one incoming joint wrench per articulation link, including the root link.
    num_envs = 2
    num_bodies = robot.num_bodies
    assert sensor.data.force.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.body_names == robot.body_names
    assert sensor.find_bodies("Arm") == ([robot.body_names.index("Arm")], ["Arm"])
    _assert_sensor_matches_physx_tensor(sensor)


def test_multi_body_articulation(sim):
    """Cartpole exposes a wrench for each link labelled by body name."""
    scene = InteractiveScene(_CartpoleSceneCfg(num_envs=2))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    num_envs = 2
    num_bodies = robot.num_bodies
    assert sensor.data.force.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.body_names == robot.body_names
    assert len(sensor.body_names) == num_bodies
    _assert_sensor_matches_physx_tensor(sensor)


def test_nested_articulation_root_resolution(sim):
    """Sensor accepts an asset prim path whose articulation root is nested in the USD asset."""
    scene = InteractiveScene(_NestedRootAntSceneCfg(num_envs=1))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    assert sensor.body_names == robot.body_names
    assert sensor.data.force.torch.shape == (1, robot.num_bodies, 3)
    assert sensor.data.torque.torch.shape == (1, robot.num_bodies, 3)
    _assert_sensor_matches_physx_tensor(sensor)


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------


def test_force_and_torque_components_at_rest(sim):
    """Component-level validation of force and torque against the PhysX tensor API."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())

    _assert_sensor_matches_physx_tensor(sensor)

    arm_idx = robot.body_names.index("Arm")
    raw_wrench = _physx_incoming_joint_wrench(sensor)
    assert torch.any(raw_wrench[:, arm_idx, :] != 0.0)


def test_non_identity_joint_frame_transform(sim):
    """PhysX raw body-frame wrench is converted to the child-side joint frame."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    _set_child_joint_frame(scene, "Arm")
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    arm_idx = robot.body_names.index("Arm")

    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())

    raw_wrench = _physx_incoming_joint_wrench(sensor)
    expected_force, expected_torque = _physx_incoming_joint_wrench_in_joint_frame(sensor, raw_wrench)
    torch.testing.assert_close(sensor.data.force.torch, expected_force)
    torch.testing.assert_close(sensor.data.torque.torch, expected_torque)

    raw_force = raw_wrench[:, arm_idx, :3]
    raw_torque = raw_wrench[:, arm_idx, 3:]
    assert not torch.allclose(sensor.data.force.torch[:, arm_idx], raw_force)
    assert not torch.allclose(sensor.data.torque.torch[:, arm_idx], raw_torque)


def test_wrench_with_external_force_and_torque(sim):
    """Full wrench validation with external force and torque applied."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    arm_idx = robot.body_names.index("Arm")

    # Apply 10 N in body-Y and 10 N·m in body-Z on the arm (matches Newton test).
    ext_force_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_force_b[:, arm_idx, 1] = 10.0
    ext_torque_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_torque_b[:, arm_idx, 2] = 10.0

    for _ in range(800):
        robot.permanent_wrench_composer.set_forces_and_torques_index(forces=ext_force_b, torques=ext_torque_b)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    _assert_sensor_matches_physx_tensor(sensor)

    raw_wrench = _physx_incoming_joint_wrench(sensor)
    assert torch.any(raw_wrench[:, arm_idx, :] != 0.0)


def test_interior_joint_wrench_at_rest(sim):
    """Interior joint wrench matches the raw PhysX incoming-joint tensor.

    The cartpole has an interior joint (``slider_to_cart``) and a terminal
    joint (``cart_to_pole``). PhysX reports one entry for every link, so this
    test compares all link entries, including the cart link controlled by the
    interior joint, against the underlying tensor API.
    """
    scene = InteractiveScene(_CartpoleDampedSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]

    for _ in range(800):
        sim.step()
        scene.update(sim.get_physics_dt())

    _assert_sensor_matches_physx_tensor(sensor)

    cart_idx = robot.body_names.index("cart")
    raw_wrench = _physx_incoming_joint_wrench(sensor)
    assert torch.any(raw_wrench[:, cart_idx, :] != 0.0)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


def test_sensor_print(sim):
    """Test that the sensor string representation works."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    sensor_str = str(sensor)
    assert "physx" in sensor_str
    assert "Joint wrench sensor" in sensor_str


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


def test_reset_zeros_buffers(sim):
    """Resetting the sensor clears the force / torque buffers."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    assert torch.any(sensor.data.force.torch != 0), "Expected non-zero data before reset"

    sensor.reset()

    # Access raw buffers to skip lazy re-population from the PhysX view on the next data read.
    force_after = wp.to_torch(sensor._data._force)
    torque_after = wp.to_torch(sensor._data._torque)
    torch.testing.assert_close(force_after, torch.zeros_like(force_after))
    torch.testing.assert_close(torque_after, torch.zeros_like(torque_after))


def test_reset_with_env_ids_only_zeros_selected_envs(sim):
    """Partial reset via env_ids should zero the selected envs and preserve the others."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=4))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    force_before = sensor.data.force.torch.clone()
    assert torch.any(force_before != 0), "Expected non-zero data before reset"

    sensor.reset(env_ids=[0, 2])

    force_after = wp.to_torch(sensor._data._force)
    torch.testing.assert_close(force_after[0], torch.zeros_like(force_after[0]))
    torch.testing.assert_close(force_after[2], torch.zeros_like(force_after[2]))
    torch.testing.assert_close(force_after[1], force_before[1])
    torch.testing.assert_close(force_after[3], force_before[3])
