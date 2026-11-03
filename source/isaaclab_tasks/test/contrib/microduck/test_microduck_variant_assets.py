# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical fidelity of the prebuilt curated-contact and roller assets.

These checks preserve independent head-shell/world-contact, body-mass, joint-limit and
free-wheel contracts. Optional source MJCF references are supplied read-only through
``MICRODUCK_MJCF_DIR`` at the immutable original pin. No source is fetched or converted.
"""

import collections
import copy

import numpy as np
import pytest
from isaaclab_newton.physics import NewtonCfg
from microduck_asset_test_utils import prebuilt_usd_path, source_mjcf_path, spawn_test_assets

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_tasks.contrib.microduck.assets import (
    MICRODUCK_ALLCOLLISIONS_CFG,
    MICRODUCK_ALLCOLLISIONS_USD_PATH,
    MICRODUCK_CFG,
    MICRODUCK_ROLLERS_CFG,
    MICRODUCK_ROLLERS_USD_PATH,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

VARIANTS = ("allcollisions", "rollers")
"""The prebuilt models this file covers. The walking model has its own suite."""

VARIANT_ASSETS = {
    "allcollisions": (MICRODUCK_ALLCOLLISIONS_USD_PATH, MICRODUCK_ALLCOLLISIONS_CFG),
    "rollers": (MICRODUCK_ROLLERS_USD_PATH, MICRODUCK_ROLLERS_CFG),
}
"""Per variant: the converted asset it reads and the configuration that spawns it."""

NUM_SERVO_JOINTS = 14
"""Joints every MicroDuck model drives, and the dimension of every MicroDuck action space.

Pinned as a literal because it is the contract, not a measurement: the roller model adds four
hinges, and the point of the servo group's ``^(?!passive_).*`` expression is that the action space
does not grow with them.
"""

NUM_JOINTS = {"allcollisions": 14, "rollers": 18}
"""Hinge count per variant.

Transcribed rather than derived so a conversion that silently drops joints cannot match the MJCF by
truncating both sides of the comparison -- which is exactly how the importer loses a joint when two
of them share a body.
"""

PASSIVE_WHEEL_JOINTS = {"passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"}
"""The roller model's undriven hinges, named so the servo expression can exclude them."""

HEAD_SHELL_MESHES = frozenset({"top_head_shell", "jaw", "bottom_head_shell"})
"""Meshes of the three colliders ``jaw_soft`` carries on both variants.

They are what lets the head reach the ground. Upstream's ``FULL_COLLISION`` config reads like it
disables them and measurably does not, so a port that "corrects" it would leave a head-rolling task
with a head that never touches anything.
"""

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
"""Upstream ``HOME_FRAME`` (STAND2), which both variants share with the walking model."""

UNLIMITED_JOINT_RANGE = 2.0 * np.pi
"""Range [rad] beyond which a joint limit cannot bind on a joint the MJCF leaves unlimited.

Newton has no unlimited revolute joint, so the wheels arrive with an enormous but finite range. A
full turn is the smallest bound that proves the limit is out of the way of a freely spinning wheel.
"""


@pytest.fixture(scope="module")
def mj_models():
    """Compile the explicit pinned source references, without generating runtime assets."""
    mujoco = pytest.importorskip("mujoco")
    filenames = {"allcollisions": "robot_allcollisions.xml", "rollers": "robot_allcollisions_rollers.xml"}
    return {
        variant: mujoco.MjModel.from_xml_path(source_mjcf_path(filename)) for variant, filename in filenames.items()
    }


@pytest.fixture(scope="module")
def mj_joints(mj_models):
    """Per variant, per hinge joint name: its MJCF limits, dynamics and whether it is limited.

    MuJoCo applies the ``chosen_actuator`` default class while compiling, so reading the compiled
    model rather than the XML text also covers the values the joints inherit -- and the wheels, which
    inherit nothing and are the reason ``limited`` is carried along.
    """
    import mujoco

    joints = {}
    for variant, model in mj_models.items():
        per_variant = {}
        for index in range(model.njnt):
            if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            dof = model.jnt_dofadr[index]
            per_variant[name] = {
                "limited": bool(model.jnt_limited[index]),
                "lower": float(model.jnt_range[index, 0]),
                "upper": float(model.jnt_range[index, 1]),
                "armature": float(model.dof_armature[dof]),
            }
        joints[variant] = per_variant
    return joints


@pytest.fixture(scope="module")
def mj_actuators(mj_models):
    """Per variant, per actuated joint name: the position actuator's force range from the MJCF."""
    import mujoco

    actuators = {}
    for variant, model in mj_models.items():
        per_variant = {}
        for index in range(model.nu):
            joint_id = model.actuator_trnid[index, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            per_variant[name] = float(model.actuator_forcerange[index, 1])
        actuators[variant] = per_variant
    return actuators


@pytest.fixture(scope="module")
def mj_world_colliders(mj_models):
    """Per variant: the meshes the MJCF lets touch the ground, counted, and their friction.

    A ground plane carries the default ``contype``/``conaffinity`` of 1, so a geom reaches world
    contact only when both of its masks share that bit. Neither variant names more than two of its
    geoms, so the colliders are counted by the mesh they are built on, which is what the conversion
    identifies them by too.
    """
    import mujoco

    colliders = {}
    for variant, model in mj_models.items():
        meshes = collections.Counter()
        frictions = set()
        for index in range(model.ngeom):
            if not (model.geom_contype[index] & 1 and model.geom_conaffinity[index] & 1):
                continue
            meshes[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[index]))] += 1
            frictions.add(tuple(float(value) for value in model.geom_friction[index]))
        assert len(frictions) == 1, "the MicroDuck world colliders share one friction triple"
        colliders[variant] = {"meshes": meshes, "friction": frictions.pop()}
    return colliders


@pytest.fixture(scope="module")
def newton_articulations():
    """Both prebuilt variants spawned on one Newton stage, bare and through their configuration.

    The bare articulations carry no actuators, so the fidelity assertions see what the USD authored
    rather than what an :class:`~isaaclab.actuators.ActuatorBaseCfg` would write over it. The
    configured ones are what a task instantiates, and spawning them here is what proves the
    configurations resolve at all. All four share one simulation because
    :class:`~isaaclab.sim.SimulationContext` is a singleton, and are offset in x so they do not
    overlap.

    The bare articulations have no initial state, and the conversion clears the root transform, so
    they spawn with their feet below the origin. That is harmless here: none of these tests steps the
    simulation, and the stage carries no ground plane.
    """
    for path, _ in VARIANT_ASSETS.values():
        prebuilt_usd_path(path)

    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0", use_newton_actuators=True, physics=NewtonCfg()))

    configs = {}
    for index, variant in enumerate(VARIANTS):
        usd_path, variant_cfg = VARIANT_ASSETS[variant]
        configs[variant] = ArticulationCfg(
            prim_path=f"/World/envs/env_0/{variant}",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            actuators={},
        )
        configured_cfg = copy.deepcopy(variant_cfg)
        configured_cfg.prim_path = f"/World/envs/env_0/{variant}_configured"
        configured_cfg.init_state.pos = (2.0 * index + 1.0, 0.0, variant_cfg.init_state.pos[2])
        configs[f"{variant}_configured"] = configured_cfg
    assets = spawn_test_assets(configs)
    articulations = {variant: (assets[variant], assets[f"{variant}_configured"]) for variant in VARIANTS}
    sim.reset()

    yield articulations, sim.stage

    sim.stop()
    sim.clear_instance()


@pytest.fixture(scope="module")
def usd_articulations(newton_articulations):
    """Per variant: the articulation loaded straight from the USD, plus the stage it spawned on."""
    articulations, stage = newton_articulations
    return {variant: (bare, stage) for variant, (bare, _) in articulations.items()}


@pytest.fixture(scope="module")
def configured_articulations(newton_articulations):
    """Per variant: the articulation loaded through its ``ArticulationCfg``."""
    return {variant: configured for variant, (_, configured) in newton_articulations[0].items()}


def _enabled_colliders(stage, prim_path: str) -> list[Usd.Prim]:
    """Return the spawned prims of one robot that can take part in world contact."""
    return [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]


def _ancestor_names(prim: Usd.Prim) -> set[str]:
    """Return the names of a prim and all of its ancestors, up to but excluding the pseudo-root."""
    names = set()
    current = prim
    while current and not current.IsPseudoRoot():
        names.add(current.GetName())
        current = current.GetParent()
    return names


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_names_match_mjcf(usd_articulations, mj_joints, variant):
    """Every MJCF hinge survives the conversion, with no extras and none silently dropped."""
    robot, _ = usd_articulations[variant]
    assert len(mj_joints[variant]) == NUM_JOINTS[variant]
    assert robot.num_joints == NUM_JOINTS[variant]
    assert set(robot.joint_names) == set(mj_joints[variant])


@pytest.mark.parametrize("variant", VARIANTS)
def test_servo_joints_stay_fourteen(usd_articulations, variant):
    """The action space does not grow with the passive hinges the roller model adds.

    The servo group selects joints with upstream's ``^(?!passive_).*``, so this is the property that
    expression exists for: 18 hinges on the roller model, 14 of them the policy's.
    """
    robot, _ = usd_articulations[variant]
    servo_joints = [name for name in robot.joint_names if not name.startswith("passive_")]
    passive_joints = set(robot.joint_names) - set(servo_joints)

    assert len(servo_joints) == NUM_SERVO_JOINTS
    assert passive_joints == (PASSIVE_WHEEL_JOINTS if variant == "rollers" else set())


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_position_limits_match_mjcf(usd_articulations, mj_joints, variant):
    """Each limited joint keeps both of its MJCF limits, and the wheels keep none that can bind."""
    robot, _ = usd_articulations[variant]
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()

    for index, name in enumerate(robot.joint_names):
        joint = mj_joints[variant][name]
        if joint["limited"]:
            assert limits[index, 0] == pytest.approx(joint["lower"], abs=1e-5), name
            assert limits[index, 1] == pytest.approx(joint["upper"], abs=1e-5), name
        else:
            # Newton has no unlimited revolute joint, so an unlimited MJCF hinge only has to arrive
            # with a range no wheel can ever reach
            assert limits[index, 0] <= -UNLIMITED_JOINT_RANGE, name
            assert limits[index, 1] >= UNLIMITED_JOINT_RANGE, name


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_armature_matches_mjcf(usd_articulations, mj_joints, variant):
    """Armature survives conversion, including the wheels' hundredfold smaller rotor inertia."""
    robot, _ = usd_articulations[variant]
    armature = robot.data.joint_armature.torch[0].cpu().numpy()
    for index, name in enumerate(robot.joint_names):
        assert armature[index] == pytest.approx(mj_joints[variant][name]["armature"], rel=1e-5), name


@pytest.mark.parametrize("variant", VARIANTS)
def test_joint_effort_limits_match_mjcf(usd_articulations, mj_actuators, variant):
    """The actuator force range survives conversion, and the undriven wheels get no clamp."""
    robot, _ = usd_articulations[variant]
    effort_limits = robot.data.joint_effort_limits.torch[0].cpu().numpy()
    assert len(mj_actuators[variant]) == NUM_SERVO_JOINTS

    for index, name in enumerate(robot.joint_names):
        if name in mj_actuators[variant]:
            assert effort_limits[index] == pytest.approx(mj_actuators[variant][name], rel=1e-5), name
        else:
            assert not np.isfinite(effort_limits[index]), name


@pytest.mark.parametrize("variant", VARIANTS)
def test_body_masses_match_mjcf(usd_articulations, mj_models, variant):
    """The converted articulation carries the MJCF's bodies and their masses.

    The masses are compared as a sorted list rather than per name: the importer renames a body whose
    name a joint also carries (``neck_pitch`` becomes ``neck_pitch_1``), so the names are not a
    common key while the mass distribution is.
    """
    robot, _ = usd_articulations[variant]
    # MuJoCo body 0 is the world body, which is not part of the articulation
    expected = np.sort(np.asarray(mj_models[variant].body_mass[1:], dtype=float))
    actual = np.sort(robot.data.body_mass.torch[0].cpu().numpy())

    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-3)
    assert float(actual.sum()) == pytest.approx(float(expected.sum()), rel=1e-3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_world_colliders_match_mjcf(usd_articulations, mj_world_colliders, variant):
    """Exactly the geoms the MJCF lets touch the ground are enabled colliders.

    The importer authors no collision groups or filtered pairs, so without the conversion's repair
    every visual shell would drag along the terrain. Both variants leave nearly all of their
    colliders in world contact, so this also pins what stays out: the trunk's ``self_collision_only``
    power support.
    """
    robot, stage = usd_articulations[variant]
    colliders = _enabled_colliders(stage, robot.cfg.prim_path)
    expected = mj_world_colliders[variant]["meshes"]

    # the MJCF barely names these geoms, so the USD names their prim after the mesh they are built on
    assert collections.Counter(prim.GetName() for prim in colliders) == expected
    assert len(colliders) == sum(expected.values())


@pytest.mark.parametrize("variant", VARIANTS)
def test_head_shells_stay_collidable(usd_articulations, variant):
    """The three ``jaw_soft`` shells reach world contact, so the head can be landed on.

    Called out separately from the collider-set comparison because it is the one collider group a
    port is tempted to remove: upstream's ``FULL_COLLISION`` reads like an allowlist that disables
    everything unnamed, and unnamed is what these three are.
    """
    robot, stage = usd_articulations[variant]
    head_shells = [
        prim
        for prim in _enabled_colliders(stage, robot.cfg.prim_path)
        if prim.GetName() in HEAD_SHELL_MESHES and "jaw_soft" in _ancestor_names(prim)
    ]
    assert {prim.GetName() for prim in head_shells} == HEAD_SHELL_MESHES


@pytest.mark.parametrize("variant", VARIANTS)
def test_world_collider_friction_matches_mjcf(usd_articulations, mj_world_colliders, variant):
    """Every world collider resolves to a physics material carrying the MJCF friction.

    On the walking model only the two soles needed the binding the importer loses to instancing.
    Here the trunk, hips, shins, head and wheels all reach the ground, so all of them do.
    """
    robot, stage = usd_articulations[variant]
    sliding, torsional, rolling = mj_world_colliders[variant]["friction"]

    for prim in _enabled_colliders(stage, robot.cfg.prim_path):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        assert material, f"no physics material bound to {prim.GetPath()}"

        material_api = UsdPhysics.MaterialAPI(material.GetPrim())
        # MuJoCo has one sliding coefficient where UsdPhysics has a static and a dynamic one
        assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(sliding, rel=1e-5)
        assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(sliding, rel=1e-5)
        assert material.GetPrim().GetAttribute("newton:torsionalFriction").Get() == pytest.approx(torsional, rel=1e-5)
        assert material.GetPrim().GetAttribute("newton:rollingFriction").Get() == pytest.approx(rolling, rel=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_root_transform_lets_the_configuration_set_the_spawn_height(usd_articulations, variant):
    """The MJCF home height is owned by the configuration, not baked into the root transform.

    The importer writes the MJCF's ``qpos0`` into the articulation root's own transform, which would
    then compose with a configuration's initial position and double the spawn height.
    """
    robot, stage = usd_articulations[variant]
    root_prim = next(
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot.cfg.prim_path))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    transform = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert transform == Gf.Matrix4d(1.0), "the articulation root must spawn at its parent's origin"


@pytest.mark.parametrize("variant", VARIANTS)
def test_floating_base(usd_articulations, variant):
    """The conversion must not weld the root to the world; both tasks need a floating base."""
    robot, _ = usd_articulations[variant]
    assert not robot.is_fixed_base


@pytest.mark.parametrize("variant", VARIANTS)
def test_cfg_drives_the_servos_and_leaves_the_wheels_free(configured_articulations, variant):
    """The configuration reuses the walking model's servo group and actuates only the servos.

    The wheels are meant to roll on the MJCF's own properties, so being left out of every actuator
    group is the intended state rather than an oversight -- the articulation only warns about it.
    """
    robot = configured_articulations[variant]
    assert set(robot.actuators) == {"servos"}

    servos = robot.cfg.actuators["servos"]
    _, servo_names = robot.find_joints(servos.joint_names_expr)
    assert len(servo_names) == NUM_SERVO_JOINTS
    assert set(servo_names) == set(robot.joint_names) - PASSIVE_WHEEL_JOINTS
    # the servo deployment itself is pinned by the walking model's suite; this pins that it is shared
    assert servos == MICRODUCK_CFG.actuators["servos"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_cfg_default_joint_pos_is_the_home_pose(configured_articulations, variant):
    """Every servo resets to its upstream ``HOME_FRAME`` value and every wheel to zero."""
    robot = configured_articulations[variant]
    default_joint_pos = robot.data.default_joint_pos.torch[0].cpu().numpy()
    assert len(HOME_POSE) == NUM_SERVO_JOINTS

    for index, name in enumerate(robot.joint_names):
        expected = 0.0 if name in PASSIVE_WHEEL_JOINTS else HOME_POSE[name]
        assert default_joint_pos[index] == pytest.approx(expected, abs=1e-6), name
