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
from isaaclab.utils import math as math_utils
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


def _compute_expected_wrench_in_joint_frame(
    sensor,
    robot,
    env: int,
    joint: int,
    gravity: torch.Tensor,
    ext_force_b: torch.Tensor | None = None,
    ext_torque_b: torch.Tensor | None = None,
    descendant_body_names: list[str] | None = None,
):
    """Compute the analytical joint-frame wrench for a single joint.

    Uses the same geometric data (body_com, joint_X_c, body_q) and frame
    transformations as the kernel, but computes the wrench analytically from
    known loads rather than reading body_parent_f.  Computes the moment of
    forces about the joint anchor and rotates the result into the child-side
    joint frame.

    For terminal links, the wrench is due to the child body alone.  For
    interior joints, pass all bodies in the subtree below the joint via
    ``descendant_body_names`` so the helper sums their gravitational
    contributions.

    Args:
        sensor: The JointWrenchSensor instance (used to read Newton model bindings).
        robot: The Articulation asset (used for body mass lookup).
        env: Environment index.
        joint: Joint index within the sensor.
        gravity: Gravity vector in world frame, shape (3,).
        ext_force_b: External force on the child body in body frame [N], shape (3,).
        ext_torque_b: External torque on the child body in body frame [N·m], shape (3,).
        descendant_body_names: Bodies whose gravitational load acts through this
            joint.  Defaults to the child body only (correct for terminal links).
            For an interior joint, pass all bodies in the subtree below the joint.

    Returns:
        A tuple of (force, torque) tensors, each shape (3,), in the child-side
        joint frame.
    """
    body_idx = wp.to_torch(sensor._joint_child)[joint].item()

    # Link transform in world (of the child body — defines the joint frame).
    link_xform = wp.to_torch(sensor._sim_bind_body_q)[env, body_idx]  # (7,) = pos(3) + quat(4)
    link_pos = link_xform[:3]
    link_quat = link_xform[3:]  # wp.quatf = (x, y, z, w)

    # Joint anchor and orientation in world = link_xform * joint_X_c.
    joint_X_c = wp.to_torch(sensor._sim_bind_joint_X_c)[env, joint]  # (7,)
    jxc_pos = joint_X_c[:3]
    jxc_quat = joint_X_c[3:]
    anchor_world = link_pos + math_utils.quat_apply(link_quat.unsqueeze(0), jxc_pos.unsqueeze(0)).squeeze(0)
    joint_quat_world = math_utils.quat_mul(link_quat.unsqueeze(0), jxc_quat.unsqueeze(0)).squeeze(0)

    # Bodies whose weight contributes to the wrench at this joint.
    if descendant_body_names is None:
        descendant_body_names = [sensor.body_names[joint]]

    link_names = list(sensor._root_view.link_names)

    total_force_w = torch.zeros(3, device=gravity.device)
    total_torque_w = torch.zeros(3, device=gravity.device)

    for body_name in descendant_body_names:
        b_idx = link_names.index(body_name)
        b_xform = wp.to_torch(sensor._sim_bind_body_q)[env, b_idx]
        b_pos = b_xform[:3]
        b_quat = b_xform[3:]
        b_com_local = wp.to_torch(sensor._sim_bind_body_com)[env, b_idx]
        b_com_world = b_pos + math_utils.quat_apply(b_quat.unsqueeze(0), b_com_local.unsqueeze(0)).squeeze(0)

        art_b_idx = robot.body_names.index(body_name)
        mass = robot.data.body_mass.torch[env, art_b_idx].item()
        weight_w = mass * gravity

        total_force_w = total_force_w + weight_w
        r = b_com_world - anchor_world
        total_torque_w = total_torque_w + torch.cross(r, weight_w, dim=-1)

    # External force/torque on the child body only (if provided).  Actuator
    # torque is intentionally omitted; see tolerance comment in calling tests.
    if ext_force_b is not None:
        ext_force_w = math_utils.quat_apply(link_quat.unsqueeze(0), ext_force_b.unsqueeze(0)).squeeze(0)
        total_force_w = total_force_w + ext_force_w
        # Moment of the external force about the joint anchor (applied at child COM).
        child_com_local = wp.to_torch(sensor._sim_bind_body_com)[env, body_idx]
        child_com_world = link_pos + math_utils.quat_apply(
            link_quat.unsqueeze(0), child_com_local.unsqueeze(0)
        ).squeeze(0)
        total_torque_w = total_torque_w + torch.cross(child_com_world - anchor_world, ext_force_w, dim=-1)
    if ext_torque_b is not None:
        total_torque_w = total_torque_w + math_utils.quat_apply(
            link_quat.unsqueeze(0), ext_torque_b.unsqueeze(0)
        ).squeeze(0)

    # Reaction wrench = negation of total wrench (joint supports against all loads).
    reaction_force_w = -total_force_w
    reaction_torque_w = -total_torque_w

    # Rotate into joint frame.
    expected_force = math_utils.quat_apply_inverse(
        joint_quat_world.unsqueeze(0), reaction_force_w.unsqueeze(0)
    ).squeeze(0)
    expected_torque = math_utils.quat_apply_inverse(
        joint_quat_world.unsqueeze(0), reaction_torque_w.unsqueeze(0)
    ).squeeze(0)

    return expected_force, expected_torque


def test_force_and_torque_components_at_rest(sim):
    """Component-level validation of force and torque against analytical expectations (gravity only)."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())

    gravity = torch.tensor(sim.cfg.gravity, device=sim.device)
    expected_force, expected_torque = _compute_expected_wrench_in_joint_frame(
        sensor,
        robot,
        env=0,
        joint=0,
        gravity=gravity,
    )

    force = sensor.data.force.torch[0, 0]
    torque = sensor.data.torque.torch[0, 0]

    torch.testing.assert_close(force, expected_force, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(torque, expected_torque, atol=1e-2, rtol=1e-3)


def test_wrench_with_external_force_and_torque(sim):
    """Full analytical wrench validation with external force and torque applied.

    Mirrors the PhysX ``test_body_incoming_joint_wrench_b_single_joint`` pattern:
    apply a known wrench, settle, compute the expected reaction wrench analytically,
    and compare component-by-component.
    """
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    arm_idx = robot.body_names.index("Arm")

    # Apply 10 N in body-Y and 10 N·m in body-Z on the arm (matches PhysX test).
    ext_force_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_force_b[:, arm_idx, 1] = 10.0
    ext_torque_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_torque_b[:, arm_idx, 2] = 10.0

    for _ in range(800):
        robot.permanent_wrench_composer.set_forces_and_torques_index(forces=ext_force_b, torques=ext_torque_b)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    gravity = torch.tensor(sim.cfg.gravity, device=sim.device)
    expected_force, expected_torque = _compute_expected_wrench_in_joint_frame(
        sensor,
        robot,
        env=0,
        joint=0,
        gravity=gravity,
        ext_force_b=ext_force_b[0, arm_idx],
        ext_torque_b=ext_torque_b[0, arm_idx],
    )

    force = sensor.data.force.torch[0, 0]
    torque = sensor.data.torque.torch[0, 0]

    # The PD actuator contributes a small torque (~0.1 N·m) to body_parent_f that is
    # not modelled in the analytical helper.  Force is unaffected (actuator is pure torque).
    torch.testing.assert_close(force, expected_force, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(torque, expected_torque, atol=0.15, rtol=1e-2)


def test_interior_joint_wrench_at_rest(sim):
    """Interior joint wrench accounts for the weight of all descendant bodies.

    The cartpole has two joints: ``slider_to_cart`` (interior, supports cart
    and pole) and ``cart_to_pole`` (terminal, supports pole only).  At steady
    state with gravity as the only load, the reaction wrench at the interior
    joint must equal the combined weight of cart and pole, with torque
    computed from each body's moment about the joint anchor.
    """
    scene = InteractiveScene(_CartpoleDampedSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]

    for _ in range(800):
        sim.step()
        scene.update(sim.get_physics_dt())

    gravity = torch.tensor(sim.cfg.gravity, device=sim.device)

    # Interior joint (index 0, slider_to_cart): reaction wrench supports
    # all bodies in the subtree — both cart and pole.
    expected_force, expected_torque = _compute_expected_wrench_in_joint_frame(
        sensor,
        robot,
        env=0,
        joint=0,
        gravity=gravity,
        descendant_body_names=list(sensor.body_names),
    )

    force = sensor.data.force.torch[0, 0]
    torque = sensor.data.torque.torch[0, 0]

    torch.testing.assert_close(force, expected_force, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(torque, expected_torque, atol=1e-2, rtol=1e-3)


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
