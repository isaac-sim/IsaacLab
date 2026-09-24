# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton JointWrenchSensor."""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sensors"))

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.sim.schemas import PhysxJointCfg
from joint_wrench_contract import test_joint_wrench_frame  # noqa: F401

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.joint_wrench import JointWrenchSensor, JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path

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
# Initialization
# ---------------------------------------------------------------------------


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
#
# The joint-frame convention (orientation and anchor) is owned by the shared ``test_joint_wrench_frame``
# contract. The tests below check load paths with frame-independent quantities from public asset data.
# ---------------------------------------------------------------------------


def test_wrench_with_external_force_and_torque(sim):
    """External loads applied through the wrench composer reach the reported joint wrench.

    The arm first settles under gravity alone, then under an additional body-frame force and torque. Force
    magnitudes are frame-independent, and so is ``dF . dtau``: with ``dF = -f`` and
    ``dtau = -(tau + r x f)`` for any anchor offset ``r``, it equals ``f . tau``.
    """
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    arm_idx = robot.body_names.index("Arm")
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device)
    weight_w = robot.data.body_mass.torch[0, arm_idx] * gravity

    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())
    force_gravity = sensor.data.force.torch[0, 0].clone()
    torque_gravity = sensor.data.torque.torch[0, 0].clone()
    torch.testing.assert_close(force_gravity.norm(), weight_w.norm(), atol=1e-2, rtol=1e-3)

    # 10 N along body Y and a body torque with a component along that force.
    ext_force_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_force_b[:, arm_idx, 1] = 10.0
    ext_torque_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_torque_b[:, arm_idx, 1] = 5.0
    ext_torque_b[:, arm_idx, 2] = 10.0

    for _ in range(800):
        robot.permanent_wrench_composer.set_forces_and_torques_index(forces=ext_force_b, torques=ext_torque_b)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    force = sensor.data.force.torch[0, 0]
    torque = sensor.data.torque.torch[0, 0]
    arm_quat_w = robot.data.body_link_quat_w.torch[0, arm_idx]
    ext_force_w = math_utils.quat_apply(arm_quat_w.unsqueeze(0), ext_force_b[0, arm_idx].unsqueeze(0)).squeeze(0)
    torch.testing.assert_close(force.norm(), (weight_w + ext_force_w).norm(), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close((force - force_gravity).norm(), ext_force_b[0, arm_idx].norm(), atol=1e-2, rtol=1e-3)
    # The PD actuator adds a small torque (~0.1 N·m) along the joint axis that is not modelled here.
    expected_dot = torch.dot(ext_force_b[0, arm_idx], ext_torque_b[0, arm_idx])
    torch.testing.assert_close(
        torch.dot(force - force_gravity, torque - torque_gravity), expected_dot, atol=1.5, rtol=0.0
    )


@pytest.mark.parametrize("fixed_pole", [False, True])
def test_interior_joint_wrench_at_rest(sim, tmp_path, fixed_pole):
    """Cart supports both masses; the pole joint supports the pole even when welded."""
    scene_cfg = _CartpoleDampedSceneCfg(num_envs=1)
    if fixed_pole:
        scene_cfg.robot.actuators.pop("pole_actuator")
        scene_cfg.robot.init_state.joint_pos.pop("cart_to_pole")
        stage = Usd.Stage.Open(retrieve_file_path(scene_cfg.robot.spawn.usd_path))
        stage.SetEditTarget(stage.GetSessionLayer())
        pole_joint = next(prim for prim in stage.Traverse() if prim.GetName() == "cart_to_pole")
        UsdPhysics.FixedJoint.Define(stage, pole_joint.GetPath())
        scene_cfg.robot.spawn.usd_path = str(tmp_path / "fixed_cartpole.usda")
        stage.Export(scene_cfg.robot.spawn.usd_path)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    assert sensor.body_names == ["cart", "pole"]
    assert sensor._root_view is robot.root_view
    assert robot.root_view.joint_names == (["slider_to_cart"] if fixed_pole else robot.joint_names)

    for _ in range(800):
        sim.step()
        scene.update(sim.get_physics_dt())

    assert sensor.data.force.torch.shape == sensor.data.torque.torch.shape == (1, 2, 3)
    # Each joint carries the weight of its subtree, whatever frame the force is expressed in.
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device)
    masses = robot.data.body_mass.torch[0]
    for joint, descendants in enumerate((["cart", "pole"], ["pole"])):
        subtree_mass = sum(masses[robot.body_names.index(name)] for name in descendants)
        torch.testing.assert_close(
            sensor.data.force.torch[0, joint].norm(), (subtree_mass * gravity).norm(), atol=1e-2, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


def test_no_stale_data_after_scene_reset(sim):
    """Regression for #4970: ``scene.reset(env_ids)`` must not surface pre-reset wrenches (Newton).

    Mirrors the PhysX equivalent. The joint-wrench sensor's lazy ``data`` accessor must not
    refetch from the Newton articulation view here (the wrench buffer reflects the previous step).
    A partial reset leaves the other envs untouched and a full sensor reset zeroes every env.
    """
    num_envs = 4
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=num_envs))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    # revolute_articulation has one joint whose child is "Arm".
    assert sensor.body_names == ["Arm"]
    assert sensor.data.force.torch.shape == sensor.data.torque.torch.shape == (num_envs, 1, 3)
    pre_reset_force = sensor.data.force.torch.clone()
    assert torch.all(torch.any(pre_reset_force != 0, dim=(1, 2))), "Expected non-zero wrench in every env"

    scene.reset(env_ids=torch.tensor([0, 2], device=sensor.device))

    post_reset_force = sensor.data.force.torch
    post_reset_torque = sensor.data.torque.torch
    for env in (0, 2):
        torch.testing.assert_close(post_reset_force[env], torch.zeros_like(post_reset_force[env]))
        torch.testing.assert_close(post_reset_torque[env], torch.zeros_like(post_reset_torque[env]))
    for env in (1, 3):
        torch.testing.assert_close(post_reset_force[env], pre_reset_force[env])

    sensor.reset()

    post_reset_force = sensor.data.force.torch
    post_reset_torque = sensor.data.torque.torch
    torch.testing.assert_close(post_reset_force, torch.zeros_like(post_reset_force))
    torch.testing.assert_close(post_reset_torque, torch.zeros_like(post_reset_torque))


@pytest.mark.parametrize("root_type", [newton.JointType.FREE, newton.JointType.FIXED, newton.JointType.REVOLUTE])
def test_fixed_joint_selection(sim, monkeypatch, root_type):
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
    model = builder.finalize(device="cpu")
    state = model.state()
    root_expr = "/World/envs/env_.*/Robot"
    views = {}
    view_factory = Mock(wraps=ArticulationView)
    monkeypatch.setattr(sensor_module, "ArticulationView", view_factory)
    monkeypatch.setattr(sensor_module.NewtonManager, "views", views)
    monkeypatch.setattr(sensor_module.NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(sensor_module.NewtonManager, "get_state_0", lambda: state)
    monkeypatch.setattr(sensor_module.BaseJointWrenchSensor, "_initialize_impl", lambda self: None)
    monkeypatch.setattr(sensor_module, "resolve_matching_prims_from_source", lambda *a, **kw: [(None, root_expr)])
    sensor = sensor_module.JointWrenchSensor(JointWrenchSensorCfg(prim_path=root_expr))
    sensor._device, sensor._num_envs = "cpu", 2
    sensor._initialize_impl()

    expected_names = ["base", "mount", "tool"] if root_type == newton.JointType.REVOLUTE else ["mount", "tool"]
    assert sensor.body_names == expected_names
    assert sensor._data._force.shape == (2, len(expected_names))
    tool_index = sensor.find_bodies("tool")[0][0]
    np.testing.assert_allclose(sensor._sim_bind_joint_X_c.numpy()[:, tool_index, 0], [0.1, 0.2])
    np.testing.assert_array_equal(
        sensor._joint_child.numpy(), [0, 2, 1] if root_type == newton.JointType.REVOLUTE else [2, 1]
    )
    # Sensing must not create a second view or include welds in the control-joint selection.
    view = views[sensor_module.NewtonManager, root_expr]
    assert sensor._root_view is view
    assert view_factory.call_count == 1
    assert view.joint_count == int(root_type == newton.JointType.REVOLUTE)
    assert len(views) == 1
    assert sensor._sim_bind_body_parent_f.ptr == view.get_attribute("body_parent_f", state).ptr
