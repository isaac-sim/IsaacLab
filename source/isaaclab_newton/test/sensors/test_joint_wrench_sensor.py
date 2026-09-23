# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton JointWrenchSensor."""

import re
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.sim.schemas import PhysxJointCfg

from pxr import Gf, Usd, UsdPhysics

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

from isaaclab_assets.robots.ant import ANT_CFG


def _make_single_joint_articulation_cfg() -> ArticulationCfg:
    """Single-joint revolute test articulation (root ``CenterPivot`` + arm ``Arm``)."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=[sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0)],
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
                joint_names_expr=["slider_to_cart"], joint_effort_limit=400.0, stiffness=0.0, damping=10.0
            ),
            "pole_actuator": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"], joint_effort_limit=400.0, stiffness=0.0, damping=pole_damping
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

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    # revolute_articulation has one joint whose child is "Arm".
    num_envs = 2
    num_joints = 1
    assert sensor.data.force.torch.shape == (num_envs, num_joints, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_joints, 3)
    assert sensor.body_names == ["Arm"]
    assert sensor._root_view is robot.root_view  # noqa: SLF001


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


def test_nested_articulation_root_resolution(sim):
    """Sensor covers a nested articulation root from the configured asset prefix."""
    scene = InteractiveScene(_NestedRootAntSceneCfg(num_envs=1))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    assert len(sensor.body_names) == robot.num_joints
    assert set(sensor.body_names).issubset(set(robot.body_names))
    assert sensor.data.force.torch.shape == (1, robot.num_joints, 3)
    assert sensor.data.torque.torch.shape == (1, robot.num_joints, 3)


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
# Reset behavior
# ---------------------------------------------------------------------------


def test_reset_zeros_selected_then_all_envs(sim):
    """Partial reset zeros only the selected envs; a full reset clears every force / torque buffer."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=4))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    force_before = sensor.data.force.torch.clone()
    assert torch.all(torch.any(force_before != 0, dim=(1, 2))), "Expected non-zero data in every env before reset"

    sensor.reset(env_ids=[0, 2])

    # Access raw buffers to skip lazy re-population from the Newton view on the next data read.
    force_after = wp.to_torch(sensor._data._force)
    torch.testing.assert_close(force_after[0], torch.zeros_like(force_after[0]))
    torch.testing.assert_close(force_after[2], torch.zeros_like(force_after[2]))
    torch.testing.assert_close(force_after[1], force_before[1])
    torch.testing.assert_close(force_after[3], force_before[3])

    sensor.reset()

    force_after = wp.to_torch(sensor._data._force)
    torque_after = wp.to_torch(sensor._data._torque)
    torch.testing.assert_close(force_after, torch.zeros_like(force_after))
    torch.testing.assert_close(torque_after, torch.zeros_like(torque_after))


def test_no_stale_data_after_scene_reset(sim):
    """Regression for #4970: ``scene.reset(env_ids)`` must not surface pre-reset wrenches (Newton).

    Mirrors the PhysX equivalent. The joint-wrench sensor's lazy ``data`` accessor must not
    refetch from the Newton articulation view here (the wrench buffer reflects the previous step).
    """
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    pre_reset_force = sensor.data.force.torch.clone()
    pre_reset_torque = sensor.data.torque.torch.clone()
    assert torch.any(pre_reset_force != 0) or torch.any(pre_reset_torque != 0), "Expected non-zero wrench before reset"

    scene.reset(env_ids=torch.tensor([0], device=sensor.device))

    post_reset_force = sensor.data.force.torch
    post_reset_torque = sensor.data.torque.torch
    torch.testing.assert_close(post_reset_force, torch.zeros_like(post_reset_force))
    torch.testing.assert_close(post_reset_torque, torch.zeros_like(post_reset_torque))


@pytest.mark.parametrize("rotated_joint_frame", [False, True])
def test_fixed_tool_wrench(sim, tmp_path, rotated_joint_frame):
    """A welded tool reports its weight and moment with the same analytic result on both backends."""
    usd_path = Path(__file__).resolve().parents[3] / "isaaclab/test/sensors/data/welded_tool.usda"
    if rotated_joint_frame:
        rotated_path = tmp_path / "rotated_tool.usda"
        rotated_path.write_text(usd_path.read_text())
        usd_path = rotated_path
        asset_stage = Usd.Stage.Open(str(usd_path))
        joint = UsdPhysics.FixedJoint.Get(asset_stage, "/Robot/wrist")
        rotation = Gf.Quatf(2.0**-0.5, 0.0, 0.0, 2.0**-0.5)
        joint.CreateLocalRot0Attr(rotation)
        joint.CreateLocalRot1Attr(rotation)
        asset_stage.GetRootLayer().Save()
    scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=3.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=["hinge"], stiffness=0.0, damping=0.0)},
    )
    scene_cfg.wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot, sensor = scene["robot"], scene["wrench"]
    assert robot.joint_names == ["hinge"]
    assert sensor.find_bodies("tool")[1] == ["tool"]
    assert sensor.body_names == ["arm", "tool"]
    assert sensor._root_view is robot.root_view
    assert robot.root_view.joint_names == ["hinge"]
    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())

    # The hinge supports both masses. The wrist supports only the tool, with a 0.15 m lever arm.
    gravity = -sim.cfg.gravity[2]
    moment = -0.5 * gravity * 0.15
    tool_torque = (moment, 0.0, 0.0) if rotated_joint_frame else (0.0, moment, 0.0)
    for body_name, mass, torque in (
        ("arm", 1.5, (0.0, -(1.0 * 0.3 + 0.5 * 0.75) * gravity, 0.0)),
        ("tool", 0.5, tool_torque),
    ):
        body_id = sensor.find_bodies(body_name)[0][0]
        expected_force = torch.tensor((0.0, 0.0, mass * gravity), device=sim.device).expand(2, -1)
        expected_torque = torch.tensor(torque, device=sim.device).expand(2, -1)
        torch.testing.assert_close(sensor.data.force.torch[:, body_id], expected_force, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(sensor.data.torque.torch[:, body_id], expected_torque, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("cached_view", [False, True])
@pytest.mark.parametrize("root_type", [newton.JointType.FREE, newton.JointType.FIXED, newton.JointType.REVOLUTE])
def test_fixed_joint_selection(sim, monkeypatch, device, cached_view, root_type):
    """Select tree welds independently of control joints, with world roots and loops excluded."""
    from isaaclab_newton.sensors.joint_wrench import joint_wrench_sensor as sensor_module
    from newton.selection import ArticulationView

    # Another articulation precedes the sensor's target in each world. The target has no
    # movable internal joints; base, mount, and tool form a tree with an additional loop weld.
    world = newton.ModelBuilder()
    unrelated = world.add_link(label="unrelated")
    world.add_articulation([world.add_joint_revolute(-1, unrelated)], label="Other")
    base = world.add_link(label="base")
    tool = world.add_link(label="tool")
    mount = world.add_link(label="mount")
    if root_type == newton.JointType.FREE:
        root_joint = world.add_joint_free(base)
    elif root_type == newton.JointType.FIXED:
        root_joint = world.add_joint_fixed(-1, base)
    else:
        root_joint = world.add_joint_revolute(-1, base)
    mount_joint = world.add_joint_fixed(base, mount, label="mount_joint")
    wrist_joint = world.add_joint_fixed(mount, tool, label="wrist")
    world.add_articulation([root_joint, mount_joint, wrist_joint], label="Robot")
    world.add_joint_fixed(tool, base, label="loop_weld")

    builder = newton.ModelBuilder()
    builder.request_state_attributes("body_parent_f")
    for env in range(2):
        # Distinct anchors catch accidental broadcasting of the first world's joint frames.
        world.joint_X_c[wrist_joint] = wp.transform(wp.vec3(0.1 * (env + 1), 0.0, 0.0), wp.quat_identity())
        builder.add_world(world, label_prefix=f"/World/envs/env_{env}")
    model = builder.finalize(device=device)
    state = model.state()
    root_expr = "/World/envs/env_.*/Robot"
    views = {}
    if cached_view:
        views[sensor_module.NewtonManager, root_expr] = ArticulationView(
            model, re.compile(root_expr), exclude_joint_types=[newton.JointType.FREE, newton.JointType.FIXED]
        )
    original_view = views.get((sensor_module.NewtonManager, root_expr))
    view_factory = Mock(wraps=ArticulationView)
    monkeypatch.setattr(sensor_module, "ArticulationView", view_factory)
    monkeypatch.setattr(sensor_module.NewtonManager, "views", views)
    monkeypatch.setattr(sensor_module.NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(sensor_module.NewtonManager, "get_state_0", lambda: state)
    monkeypatch.setattr(sensor_module.BaseJointWrenchSensor, "_initialize_impl", lambda self: None)
    monkeypatch.setattr(sensor_module, "resolve_matching_prims_from_source", lambda *a, **kw: [(None, root_expr)])
    sensor = sensor_module.JointWrenchSensor(JointWrenchSensorCfg(prim_path=root_expr))
    sensor._device, sensor._num_envs = device, 2
    sensor._initialize_impl()

    expected_names = ["base", "mount", "tool"] if root_type == newton.JointType.REVOLUTE else ["mount", "tool"]
    assert sensor.body_names == expected_names
    assert sensor._data._force.shape == (2, len(expected_names))
    tool_index = sensor.find_bodies("tool")[0][0]
    np.testing.assert_allclose(sensor._sim_bind_joint_X_c.numpy()[:, tool_index, 0], [0.1, 0.2])
    np.testing.assert_array_equal(
        sensor._joint_child.numpy(), [0, 2, 1] if root_type == newton.JointType.REVOLUTE else [2, 1]
    )
    # The sensor must neither replace the cached view nor add welds to its control-joint selection.
    view = views[sensor_module.NewtonManager, root_expr]
    assert sensor._root_view is view
    if cached_view:
        assert view is original_view
    assert view_factory.call_count == int(not cached_view)
    assert view.joint_count == int(root_type == newton.JointType.REVOLUTE)
    assert len(views) == 1
    assert sensor._sim_bind_body_parent_f.ptr == view.get_attribute("body_parent_f", state).ptr
