# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the imported contacts, passive joints, and native gearbox encoder.

The old model suites never step a serial-play robot. In particular, summing only
the policy observations cannot pass the small-signal firmware torque check here.
"""

import copy
import os
import re

os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from microduck_asset_test_utils import prebuilt_usd_path, spawn_test_assets

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_tasks.contrib.microduck import assets as microduck

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

_CONFIGS = {
    "fullcontact": microduck.MICRODUCK_FULLCONTACT_CFG,
    "fullcontact_backlash": microduck.MICRODUCK_FULLCONTACT_BACKLASH_CFG,
    "walk_backlash": microduck.MICRODUCK_WALK_BACKLASH_CFG,
    "groundcontact_backlash": microduck.MICRODUCK_GROUNDCONTACT_BACKLASH_CFG,
    "rollers_backlash": microduck.MICRODUCK_ROLLERS_BACKLASH_CFG,
}


@pytest.fixture(scope="module")
def asset_scene():
    for cfg in _CONFIGS.values():
        prebuilt_usd_path(cfg.spawn.usd_path)
    sim_utils.create_new_stage()
    sim = SimulationContext(
        SimulationCfg(
            dt=0.005,
            device="cuda:0",
            use_newton_actuators=True,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=2048, nconmax=1024)),
        )
    )
    cfgs = {}
    for index, (name, original) in enumerate(_CONFIGS.items()):
        cfg = copy.deepcopy(original)
        cfg.prim_path = f"/World/envs/env_0/{name}"
        cfg.init_state.pos = (index * 2.0, 0.0, 0.14 if name == "rollers_backlash" else 0.125)
        # Remove stochastic supply and delay only in the diagnostic fixture.
        servo = cfg.actuators["servos"]
        servo.vin = 8.2
        servo.vin_range = None
        servo.vin_drop_gain_range = None
        servo.min_delay = servo.max_delay = 0
        cfgs[name] = cfg
    floor = RigidObjectCfg(
        prim_path="/World/envs/env_0/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(24.0, 4.0, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, -0.05)),
    )
    assets = spawn_test_assets({**cfgs, "floor": floor})
    robots = {name: assets[name] for name in cfgs}
    sim.reset()
    for robot in robots.values():
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_pos.torch))
    sim.forward()
    for robot in robots.values():
        robot.update(0.0)
    sim._app_control_on_stop_handle = None
    yield robots, sim
    sim.stop()
    sim.clear_instance()


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_passive_coordinates_survive_and_original_bodies_keep_mass(asset_scene, name):
    robots, _ = asset_scene
    robot = robots[name]
    backlash = name.endswith("_backlash")
    assert robot.num_joints == (32 if name == "rollers_backlash" else 28 if backlash else 14)
    servo_names = [n for n in robot.joint_names if not n.startswith("passive_")]
    assert len(servo_names) == 14
    passive = [f"passive_{n}_backlash" for n in servo_names] if backlash else []
    assert set(passive).issubset(robot.joint_names)
    dummy_ids = [i for i, n in enumerate(robot.body_names) if n.endswith("_backlash_dummy")]
    assert len(dummy_ids) == (14 if backlash else 0)
    if backlash:
        np.testing.assert_allclose(robot.data.body_mass.torch[0, dummy_ids].cpu(), 1e-6, rtol=1e-6)
        ids, _ = robot.find_joints(passive, preserve_order=True)
        np.testing.assert_allclose(
            robot.data.joint_pos_limits.torch[0, ids].cpu(), [[-np.pi / 180, np.pi / 180]] * 14, atol=1e-7
        )
        np.testing.assert_allclose(robot.data.joint_armature.torch[0, ids].cpu(), 0.001, rtol=1e-6)
        np.testing.assert_allclose(robot.data.joint_viscous_friction_coeff.torch[0, ids].cpu(), 0.01, rtol=1e-5)
        backend = asset_scene[1].backend_manager.backend
        model, solver = backend.model, backend.solver
        dof_map = solver.mjc_dof_to_newton_dof.numpy()[0]
        for index, label in enumerate(model.joint_label):
            if not label.startswith(robot.cfg.prim_path + "/") or label.rsplit("/", 1)[-1] not in passive:
                continue
            mj_dof = int(np.flatnonzero(dof_map == model.joint_qd_start.numpy()[index])[0])
            mj_joint = int(solver.mj_model.dof_jntid[mj_dof])
            np.testing.assert_allclose(solver.mj_model.jnt_solref[mj_joint], [0.01, 1.0], rtol=1e-6)
            np.testing.assert_allclose(solver.mj_model.jnt_solimp[mj_joint], [0.95, 0.999, 0.0001, 0.5, 2.0], rtol=1e-6)

    physical_mass = robot.data.body_mass.torch[0].sum() - len(dummy_ids) * 1e-6
    expected = 0.73682964 if name == "rollers_backlash" else 0.73724318
    assert float(physical_mass) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", ["fullcontact", "fullcontact_backlash"])
def test_fullcontact_properties_reach_the_imported_model(asset_scene, name):
    robots, sim = asset_scene
    robot = robots[name]
    stage = sim.stage
    colliders = [
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(robot.cfg.prim_path), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
    ]
    assert len(colliders) == 69
    housings = [p for p in colliders if p.GetParent().GetName().endswith("_servo_collision")]
    assert len(housings) == 15
    assert all(p.GetAttribute("mjc:condim").Get() == 1 for p in housings)
    feet = [p for p in colliders if p.GetParent().GetName() in {"left_foot_collision", "right_foot_collision"}]
    assert len(feet) == 2
    assert all(p.GetAttribute("mjc:priority").Get() == 1 for p in feet)
    neck = next(
        p
        for p in Usd.PrimRange(stage.GetPrimAtPath(robot.cfg.prim_path))
        if p.GetName() == "neck_pitch_1" and p.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    targets = UsdPhysics.FilteredPairsAPI(neck).GetFilteredPairsRel().GetTargets()
    assert any(p.name == "jaw_soft" for p in targets)
    # Test the compiled contact model as well as the authoring layer.
    backend = sim.backend_manager.backend
    model, solver = backend.model, backend.solver
    geom_shapes = solver.mjc_geom_to_newton_shape.numpy()[0]
    geoms = [
        i
        for i, shape in enumerate(geom_shapes)
        if shape >= 0
        and model.shape_label[shape].startswith(robot.cfg.prim_path + "/")
        and (solver.mj_model.geom_contype[i] or solver.mj_model.geom_conaffinity[i])
    ]
    assert len(geoms) == 69
    np.testing.assert_array_equal(np.sort(solver.mj_model.geom_condim[geoms]), [1] * 15 + [3] * 54)
    np.testing.assert_array_equal(np.sort(solver.mj_model.geom_priority[geoms]), [0] * 67 + [1] * 2)
    for key, count in (("COLLIDER", 69), ("SERVO", 15), ("HEAD", 30), ("TRUNK", 9)):
        pattern = getattr(microduck, f"MICRODUCK_FULLCONTACT_{key}_SHAPE_EXPR")
        pattern = pattern.replace("{ENV_REGEX_NS}/Robot", robot.cfg.prim_path)
        selected = {
            model.shape_label[geom_shapes[i]] for i in geoms if re.fullmatch(pattern, model.shape_label[geom_shapes[i]])
        }
        assert len(selected) == count
    # MuJoCo compiles the USD body-level filtered pair into its exclusion table.
    body_map = solver.mjc_body_to_newton.numpy()[0]
    excluded = {}
    for mj_body, newton_body in enumerate(body_map):
        if newton_body >= 0:
            label = model.body_label[newton_body]
            if label.startswith(robot.cfg.prim_path + "/"):
                excluded[label.rsplit("/", 1)[-1]] = mj_body
    first, second = sorted((excluded["neck_pitch_1"], excluded["jaw_soft"]))
    assert (first << 16) + second in solver.mj_model.exclude_signature


def _command_from_state(robot, sim, servo_angle, play_angle, error):
    servo_ids, servo_names = robot.find_joints(["^(?!passive_).*"], preserve_order=True)
    play_ids, _ = robot.find_joints([f"passive_{n}_backlash" for n in servo_names], preserve_order=True)
    position = robot.data.default_joint_pos.torch.clone()
    position[:, servo_ids] += servo_angle
    position[:, play_ids] = play_angle
    robot.write_joint_position_to_sim_index(position=position)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position))
    robot.actuators.reset()
    targets = position.clone()
    targets[:, servo_ids] = robot.data.default_joint_pos.torch[:, servo_ids] + error
    robot.actuators.target_command.set_position_index(value=targets)
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())
    return read_group_parameter(robot.actuators, "servos", "drive", "motor_torque").cpu().numpy().copy(), servo_ids


def test_native_firmware_uses_the_output_encoder_and_electrical_clamp(asset_scene):
    robots, sim = asset_scene
    robot = robots["fullcontact_backlash"]
    first, ids = _command_from_state(robot, sim, 0.0, 0.01, 0.1)
    equivalent, _ = _command_from_state(robot, sim, 0.01, 0.0, 0.1)
    np.testing.assert_allclose(first, equivalent, atol=2e-7, rtol=2e-5)
    motor = robot.cfg.actuators["servos"].motor
    expected = motor.kt / motor.resistance * 8.2 * 200 * motor.error_gain * 0.09
    np.testing.assert_allclose(first, expected, atol=2e-7, rtol=2e-5)
    saturated, _ = _command_from_state(robot, sim, 0.0, 0.0, 10.0)
    limit = 8.2 * motor.kt / motor.resistance
    np.testing.assert_allclose(saturated, limit, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(robot.actuators.applied_effort.torch[0, ids].cpu(), limit, atol=1e-6, rtol=1e-6)
    assert limit > 0.96
    backend = sim.backend_manager.backend
    model, solver = backend.model, backend.solver
    dof_map = solver.mjc_dof_to_newton_dof.numpy()[0]
    servo_dofs = []
    for index, label in enumerate(model.joint_label):
        if not label.startswith(robot.cfg.prim_path + "/") or label.rsplit("/", 1)[-1] not in robot.joint_names:
            continue
        if label.rsplit("/", 1)[-1].startswith("passive_"):
            continue
        mj_dof = int(np.flatnonzero(dof_map == model.joint_qd_start.numpy()[index])[0])
        servo_dofs.append(mj_dof)
        mj_joint = int(solver.mj_model.dof_jntid[mj_dof])
        np.testing.assert_allclose(solver.mj_model.jnt_actfrcrange[mj_joint], [-limit, limit], rtol=1e-6)
    assert len(servo_dofs) == 14
    np.testing.assert_allclose(solver.mjw_data.qfrc_applied.numpy()[0, servo_dofs], limit, rtol=1e-6)


def test_serial_hinges_step_finitely_and_passive_axes_are_unactuated(asset_scene):
    robots, sim = asset_scene
    for robot in robots.values():
        robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch)
        robot.write_data_to_sim()
    peak_contacts = 0
    for _ in range(32):
        sim.step(render=False)
        peak_contacts = max(peak_contacts, int(sim.backend_manager.backend.solver.mjw_data.nacon.numpy().sum()))
        for robot in robots.values():
            robot.update(sim.get_physics_dt())
            assert torch.isfinite(robot.data.joint_pos.torch).all()
            ids = [i for i, n in enumerate(robot.joint_names) if n.startswith("passive_")]
            assert torch.count_nonzero(robot.actuators.applied_effort.torch[:, ids]) == 0
    assert peak_contacts > 0
