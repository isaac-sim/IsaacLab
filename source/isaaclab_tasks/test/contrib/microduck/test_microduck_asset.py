# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical fidelity of the prebuilt walking USD and its native BAM configuration.

The runtime asset is resolved through Isaac Lab's native asset cache. Source comparisons
read an optional explicit ``MICRODUCK_MJCF_PATH`` or ``MICRODUCK_MJCF_DIR`` fixture from
the immutable original upstream pin; they never fetch source or convert an MJCF.
"""

import copy
import os

import numpy as np
import pytest
from isaaclab_newton.physics import NewtonCfg
from microduck_asset_test_utils import prebuilt_usd_path, source_mjcf_path, spawn_test_assets

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_CFG, MICRODUCK_USD_PATH

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

NUM_ACTUATED_JOINTS = 14
"""Joint count ``robot_walk.xml`` is known to have, guarding against a partial import that still
matches the MJCF because both sides were read from the same truncated model."""

HOME_POSE = {
    "left_hip_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579,
    "right_hip_pitch": 0.4579,
    "left_knee": -0.0049,
    "right_knee": 0.0049,
    "left_ankle": 0.4530,
    "right_ankle": -0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}
"""Upstream ``HOME_FRAME`` (STAND2), transcribed per joint name from ``microduck_constants.py``.

The configuration expresses it with regular expressions, so spelling it out one joint at a time is
what makes the check meaningful: a pattern that matches nothing leaves the joint at zero.
"""

UPSTREAM_RESET_HEIGHT_RANGE = (0.12, 0.13)
"""Base height [m] upstream's ``reset_base`` event samples, from ``microduck_rl``."""


@pytest.fixture(scope="module")
def mjcf_path():
    """Read the explicit pinned walking reference without fetching or conversion."""
    return source_mjcf_path("robot_walk.xml")


@pytest.fixture(scope="module")
def mj_model(mjcf_path):
    """The source MJCF, compiled by MuJoCo, used as the fidelity reference."""
    mujoco = pytest.importorskip("mujoco")
    if not os.path.isfile(mjcf_path):
        pytest.skip(f"MicroDuck MJCF checkout not available: {mjcf_path}")
    return mujoco.MjModel.from_xml_path(mjcf_path)


@pytest.fixture(scope="module")
def mj_joints(mj_model):
    """Per hinge joint name: its position limits [rad] and the dynamics MuJoCo resolved for it.

    MuJoCo applies the ``chosen_actuator`` default class while compiling, so reading the compiled
    model rather than the XML text also covers the values the joints inherit.
    """
    import mujoco

    joints = {}
    for index in range(mj_model.njnt):
        if mj_model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, index)
        dof = mj_model.jnt_dofadr[index]
        joints[name] = {
            "lower": float(mj_model.jnt_range[index, 0]),
            "upper": float(mj_model.jnt_range[index, 1]),
            "armature": float(mj_model.dof_armature[dof]),
            "damping": float(mj_model.dof_damping[dof]),
            "frictionloss": float(mj_model.dof_frictionloss[dof]),
        }
    return joints


@pytest.fixture(scope="module")
def mj_actuators(mj_model):
    """Per actuated joint name: the position actuator's gains and force range from the MJCF."""
    import mujoco

    actuators = {}
    for index in range(mj_model.nu):
        joint_id = mj_model.actuator_trnid[index, 0]
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        # MuJoCo's "position" actuator encodes gainprm=[kp, 0, ...], biasprm=[0, -kp, -kv, ...].
        actuators[name] = {
            "stiffness": float(mj_model.actuator_gainprm[index, 0]),
            "damping": float(-mj_model.actuator_biasprm[index, 2]),
            "effort_limit": float(mj_model.actuator_forcerange[index, 1]),
        }
    return actuators


@pytest.fixture(scope="module")
def mj_world_colliders(mj_model):
    """The MJCF geoms that can touch the ground, and the friction MuJoCo resolved for them.

    A ground plane carries the default ``contype``/``conaffinity`` of 1, so a geom reaches world
    contact only when both of its masks share that bit. The ``self_collision_only`` geoms use 2 and
    are excluded, which is exactly the distinction the conversion has to reproduce.
    """
    import mujoco

    colliders = {}
    for index in range(mj_model.ngeom):
        if not (mj_model.geom_contype[index] & 1 and mj_model.geom_conaffinity[index] & 1):
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, index)
        sliding, torsional, rolling = (float(value) for value in mj_model.geom_friction[index])
        colliders[name] = {"sliding": sliding, "torsional": torsional, "rolling": rolling}
    return colliders


@pytest.fixture(scope="module")
def newton_articulations():
    """The prebuilt USD spawned twice on one Newton stage: bare, and through ``MICRODUCK_CFG``.

    The bare articulation carries no actuators, so the fidelity assertions see what the USD authored
    rather than what an :class:`~isaaclab.actuators.ActuatorBaseCfg` would write over it. The
    configured one is what a task instantiates. Both share one simulation because
    :class:`~isaaclab.sim.SimulationContext` is a singleton; the configured robot is offset in x so
    the two do not overlap.

    The bare articulation has no initial state, and the conversion clears the root transform, so it
    spawns with its feet below the origin. That is harmless here: none of these tests steps the
    simulation, and the stage carries no ground plane.
    """
    prebuilt_usd_path(MICRODUCK_USD_PATH)

    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0", use_newton_actuators=True, physics=NewtonCfg()))
    bare_cfg = ArticulationCfg(
        prim_path="/World/envs/env_0/Robot", spawn=sim_utils.UsdFileCfg(usd_path=MICRODUCK_USD_PATH), actuators={}
    )
    configured_cfg = copy.deepcopy(MICRODUCK_CFG)
    configured_cfg.prim_path = "/World/envs/env_0/MicroDuck"
    configured_cfg.init_state.pos = (1.0, 0.0, MICRODUCK_CFG.init_state.pos[2])
    assets = spawn_test_assets({"bare": bare_cfg, "configured": configured_cfg})
    bare, configured = assets["bare"], assets["configured"]
    sim.reset()

    yield bare, configured, sim.stage

    sim.stop()
    sim.clear_instance()


@pytest.fixture(scope="module")
def usd_articulation(newton_articulations):
    """The articulation loaded straight from the USD, plus the stage it was spawned on."""
    bare, _, stage = newton_articulations
    return bare, stage


@pytest.fixture(scope="module")
def microduck_articulation(newton_articulations):
    """The articulation loaded through :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_CFG`."""
    return newton_articulations[1]


def test_actuated_joint_names_match_mjcf(usd_articulation, mj_joints):
    """Every MJCF hinge joint survives the conversion, with no extras."""
    robot, _ = usd_articulation
    assert len(mj_joints) == NUM_ACTUATED_JOINTS
    assert set(robot.joint_names) == set(mj_joints)


def test_joint_position_limits_match_mjcf(usd_articulation, mj_joints):
    """Each joint keeps both of its MJCF limits, which are asymmetric for the head joints."""
    robot, _ = usd_articulation
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
    for index, name in enumerate(robot.joint_names):
        assert limits[index, 0] == pytest.approx(mj_joints[name]["lower"], abs=1e-5)
        assert limits[index, 1] == pytest.approx(mj_joints[name]["upper"], abs=1e-5)


def test_joint_armature_matches_mjcf(usd_articulation, mj_joints):
    """Armature survives conversion, so the task inherits the MJCF rotor inertia."""
    robot, _ = usd_articulation
    armature = robot.data.joint_armature.torch[0].cpu().numpy()
    for index, name in enumerate(robot.joint_names):
        assert armature[index] == pytest.approx(mj_joints[name]["armature"], rel=1e-5)


def test_joint_effort_limits_match_mjcf(usd_articulation, mj_actuators):
    """The actuator force range survives conversion as the joint effort limit."""
    robot, _ = usd_articulation
    effort_limits = robot.data.joint_effort_limits.torch[0].cpu().numpy()
    for index, name in enumerate(robot.joint_names):
        assert effort_limits[index] == pytest.approx(mj_actuators[name]["effort_limit"], rel=1e-5)


def test_prebuilt_bam_friction_seed_and_zero_implicit_pd_gains(usd_articulation, mj_joints, mj_actuators):
    """The prebuilt USD seeds identified BAM friction without adding an implicit PD drive."""
    robot, _ = usd_articulation
    stiffness = robot.data.joint_stiffness.torch[0].cpu().numpy()
    damping = robot.data.joint_damping.torch[0].cpu().numpy()
    friction = robot.data.joint_friction_coeff.torch[0].cpu().numpy()

    # the MJCF values that the asset does not carry, guarding against a no-op comparison
    assert all(mj_actuators[name]["stiffness"] > 0.0 for name in robot.joint_names)
    assert all(mj_joints[name]["damping"] > 0.0 for name in robot.joint_names)
    assert all(mj_joints[name]["frictionloss"] > 0.0 for name in robot.joint_names)

    assert np.all(stiffness == 0.0)
    assert np.all(damping == 0.0)
    # The hosted BAM asset uses the identified m6 value rather than the MJCF's rounded 0.0048.
    # Native BAM publishes the live friction budget separately on every solver step.
    np.testing.assert_allclose(friction, 0.004771183165566, rtol=1e-5)


def test_total_mass_matches_mjcf(usd_articulation, mj_model):
    """The converted articulation carries the same total mass as the MJCF."""
    robot, _ = usd_articulation
    expected_mass = float(np.sum(mj_model.body_mass))
    assert float(robot.data.body_mass.torch[0].sum()) == pytest.approx(expected_mass, rel=0.01)


def test_root_transform_lets_the_configuration_set_the_spawn_height(usd_articulation, mj_model):
    """The MJCF home height is owned by the configuration, not baked into the root transform.

    The importer writes the MJCF's ``qpos0`` into the articulation root's own transform. Spawning
    applies a configuration's initial position to the prim the asset is referenced under, so a baked
    transform composes with it and doubles the spawn height. The conversion clears it; the height
    moves to :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_CFG`, which must still cover the MJCF's.
    """
    robot, stage = usd_articulation
    root_prim = next(
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot.cfg.prim_path))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    transform = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert transform == Gf.Matrix4d(1.0), "the articulation root must spawn at its parent's origin"

    low, high = UPSTREAM_RESET_HEIGHT_RANGE
    assert low <= float(mj_model.qpos0[2]) <= high
    assert MICRODUCK_CFG.init_state.pos[2] == pytest.approx((low + high) / 2.0)


def test_floating_base(usd_articulation):
    """The conversion must not weld the root to the world; the walk task needs a floating base."""
    robot, _ = usd_articulation
    assert not robot.is_fixed_base


def _enabled_colliders(stage, prim_path) -> list[Usd.Prim]:
    """Return the spawned prims that can take part in world contact."""
    return [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]


def test_world_colliders_match_mjcf(usd_articulation, mj_world_colliders):
    """Only the geoms the MJCF lets touch the ground are enabled colliders.

    The importer authors no collision groups or filtered pairs, so without the conversion's repair
    the ``self_collision_only`` shells would drag along the terrain.
    """
    robot, stage = usd_articulation
    colliders = _enabled_colliders(stage, robot.cfg.prim_path)

    assert len(mj_world_colliders) == 2, "the MJCF's world-contact set is the two foot soles"
    assert len(colliders) == len(mj_world_colliders)
    # the MJCF geoms are named '<side>_foot_collision'; the USD keeps that name on the parent Xform
    assert {prim.GetParent().GetName() for prim in colliders} == set(mj_world_colliders)


def test_foot_friction_matches_mjcf(usd_articulation, mj_world_colliders):
    """Each foot collider resolves to a physics material carrying the MJCF friction."""
    robot, stage = usd_articulation

    for prim in _enabled_colliders(stage, robot.cfg.prim_path):
        expected = mj_world_colliders[prim.GetParent().GetName()]
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        assert material, f"no physics material bound to {prim.GetPath()}"

        material_api = UsdPhysics.MaterialAPI(material.GetPrim())
        # MuJoCo has one sliding coefficient where UsdPhysics has a static and a dynamic one
        assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(expected["sliding"], rel=1e-5)
        assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(expected["sliding"], rel=1e-5)
        torsional = material.GetPrim().GetAttribute("newton:torsionalFriction").Get()
        rolling = material.GetPrim().GetAttribute("newton:rollingFriction").Get()
        assert torsional == pytest.approx(expected["torsional"], rel=1e-5)
        assert rolling == pytest.approx(expected["rolling"], rel=1e-5)


def test_microduck_cfg_actuates_every_joint(microduck_articulation):
    """The servo group's expression reaches all 14 joints and no joint is left undriven."""
    robot = microduck_articulation
    assert set(robot.actuators) == {"servos"}

    servos = robot.cfg.actuators["servos"]
    _, servo_names = robot.find_joints(servos.joint_names_expr)
    assert len(servo_names) == NUM_ACTUATED_JOINTS
    assert set(servo_names) == set(robot.joint_names)


def test_microduck_cfg_default_joint_pos_is_the_home_pose(microduck_articulation):
    """Every joint resets to its upstream ``HOME_FRAME`` value."""
    robot = microduck_articulation
    default_joint_pos = robot.data.default_joint_pos.torch[0].cpu().numpy()
    assert len(HOME_POSE) == NUM_ACTUATED_JOINTS

    for index, name in enumerate(robot.joint_names):
        assert default_joint_pos[index] == pytest.approx(HOME_POSE[name], abs=1e-6), name


def test_microduck_cfg_restores_the_joint_dynamics_the_asset_drops(microduck_articulation, mj_joints):
    """The native drive supplies the identified damping while retaining MJCF armature.

    BAM publishes its live dry-friction budget directly to MJWarp; the generic joint-friction
    property is an authored seed and does not report that budget.
    """
    robot = microduck_articulation
    servos = robot.cfg.actuators["servos"]
    viscous = robot.data.joint_viscous_friction_coeff.torch[0].cpu().numpy()
    armature = robot.data.joint_armature.torch[0].cpu().numpy()

    # the MJCF value is ten times the fit, so a revert to it cannot pass the comparison below
    assert all(mj_joints[name]["damping"] > 5.0 * servos.motor.friction_viscous for name in robot.joint_names)

    for index, name in enumerate(robot.joint_names):
        # rel=1e-3 because the configured constant is the fit rounded to three significant digits
        assert viscous[index] == pytest.approx(servos.motor.friction_viscous, rel=1e-3), name
        # ... and the dry friction the solver gives up is the model's own: the MJCF's frictionloss
        # is the vendored fit's unloaded friction budget, which the BAM model applies per step
        assert mj_joints[name]["frictionloss"] == pytest.approx(servos.motor.friction_base, rel=1e-2), name
        # armature is left to the USD, which carries the MJCF value unchanged
        assert armature[index] == pytest.approx(mj_joints[name]["armature"], rel=1e-4), name


def test_microduck_cfg_servo_model_is_upstreams_bam_deployment(microduck_articulation, mj_actuators):
    """The servo group is the BAM model at upstream's deployment settings, cross-checked on the MJCF.

    Upstream's ``_BAM_ACTUATOR_KWARGS`` (reference section 6) are pinned field by field, because a
    silent change to any of them retrains a different robot. The gains are then cross-checked
    against the MJCF: the small-signal stiffness of the firmware loop at the middle of the battery
    range, ``kp_fw * error_gain * vin * kt / R``, is an independent re-derivation of the ``kp``
    the MJCF's fallback ``position`` actuator declares. The model's own effort limit is upstream's
    electrical stall torque at the top of the battery range, which sits *above* the MJCF's
    ``forcerange`` -- so the solver clamp, which conversion carries unchanged, binds first.
    """
    robot = microduck_articulation
    servos = robot.cfg.actuators["servos"]
    cfg = servos
    params = servos.motor
    from isaaclab.actuators.newton.adapter import read_group_parameter

    max_effort = read_group_parameter(robot.actuators, "servos", "drive", "max_effort")
    solver_limit = robot.data.joint_effort_limits.torch[0].cpu().numpy()

    assert cfg.kp_fw == 200.0
    assert cfg.vin_range == (6.5, 8.2)
    assert cfg.vin_drop_gain_range == (0.0, 0.2)
    assert cfg.vin_min == 6.0
    assert (cfg.min_delay, cfg.max_delay) == (3, 6)
    # unused by this model, and set to anything but None they warn
    assert cfg.stiffness is None and cfg.damping is None

    nominal_vin = sum(cfg.vin_range) / 2.0
    small_signal_stiffness = cfg.kp_fw * params.error_gain * nominal_vin * params.kt / params.resistance
    stall_torque = max(cfg.vin_range) * params.kt / params.resistance
    for index, name in enumerate(robot.joint_names):
        authored = mj_actuators[name]["effort_limit"]
        assert small_signal_stiffness == pytest.approx(mj_actuators[name]["stiffness"], rel=0.01), name
        assert float(max_effort[0, index]) == pytest.approx(stall_torque, rel=1e-6), name
        assert float(max_effort[0, index]) > authored, name
        assert solver_limit[index] == pytest.approx(authored, rel=1e-5), name
