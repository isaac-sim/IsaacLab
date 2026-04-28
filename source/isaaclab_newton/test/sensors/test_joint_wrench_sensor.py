# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton JointWrenchSensor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.joint_wrench import JointWrenchSensor, JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


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


def _make_cartpole_articulation_cfg() -> ArticulationCfg:
    """Two-joint cartpole articulation (cart + pole)."""
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
                joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
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


@pytest.fixture
def sim():
    """Simulation context using the Newton backend."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 200.0,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(),
            num_substeps=1,
        ),
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None
        yield sim_ctx


# ---------------------------------------------------------------------------
# Sensor data — pre-init contract
# ---------------------------------------------------------------------------


def test_data_before_init_is_none():
    """``force``/``torque`` return ``None`` before :meth:`create_buffers` runs."""
    from isaaclab_newton.sensors.joint_wrench import JointWrenchSensorData

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

    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    # revolute_articulation has one joint whose child is "Arm".
    num_envs = 2
    num_joints = 1
    assert sensor.data.force.torch.shape == (num_envs, num_joints, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_joints, 3)
    assert sensor.body_names == ["Arm"]


def test_multi_body_articulation(sim):
    """Cartpole (2 joints) exposes a wrench for each joint labelled by its child body."""
    scene = InteractiveScene(_CartpoleSceneCfg(num_envs=2))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    num_envs = 2
    num_joints = 2
    assert sensor.data.force.torch.shape == (num_envs, num_joints, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_joints, 3)
    assert len(sensor.body_names) == 2
    assert "rail" not in [n.lower() for n in sensor.body_names]


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------


def test_force_magnitude_matches_weight_at_rest(sim):
    """At steady state, |force| on the arm joint should be close to its weight."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    # PD control damps out residual oscillation within a few hundred steps.
    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())

    force = sensor.data.force.torch[0, 0]  # (3,)
    torque = sensor.data.torque.torch[0, 0]

    assert torch.isfinite(force).all(), f"Force contains non-finite values: {force}"
    assert torch.isfinite(torque).all(), f"Torque contains non-finite values: {torque}"

    # The arm COM is offset from the joint anchor, so gravity creates a non-zero
    # moment about the joint.  A zero torque would indicate a broken torque path.
    torque_mag = torque.norm().item()
    assert torque_mag > 0.1, f"|torque|={torque_mag:.3f} N·m, expected non-trivial (>0.1)"

    # Frame-independent check: at rest the only external effect is gravity on the arm.
    arm_idx = robot.body_names.index("Arm")
    arm_mass = wp.to_torch(robot.data.body_mass)[0, arm_idx].item()
    expected_weight = arm_mass * 9.81
    assert abs(force.norm().item() - expected_weight) < 0.1 * expected_weight + 0.5, (
        f"|force|={force.norm().item():.3f} N, expected ~{expected_weight:.3f} N"
    )


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


def test_sensor_print(sim):
    """Test that the sensor string representation works."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    sensor_str = str(sensor)
    assert "newton" in sensor_str
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

    # Access raw buffers to skip lazy re-population from the Newton view on the next data read.
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
