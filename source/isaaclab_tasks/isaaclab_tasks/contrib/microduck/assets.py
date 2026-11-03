# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task deployment clones, contact-site metadata and the MicroDuck floorball.

Robot geometry comes exclusively from the eight prebuilt assets in ``isaaclab_assets``. The audited
Pollen recipes retain their explicit electrical effort ceiling and identified viscous damping in
private clones; the original contributed walking configurations keep their established defaults.
All robot variants use the Isaac Lab asset server, including both full-contact variants.
"""

from pxr import Gf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import clone as clone_cfg

from isaaclab_assets.robots import microduck as robots

MICRODUCK_SERVO_VIN_RANGE = (6.5, 8.2)
"""Per-robot battery voltage range [V]."""
MICRODUCK_JOINT_DAMPING = robots.MICRODUCK_CFG.actuators["servos"].motor.friction_viscous
"""Identified XL330 viscous joint damping [N m s/rad]."""
_motor = robots.MICRODUCK_CFG.actuators["servos"].motor
MICRODUCK_SERVO_EFFORT_LIMIT = max(MICRODUCK_SERVO_VIN_RANGE) * _motor.kt / _motor.resistance
"""Electrical ceiling [N m]; the separate 1.75 A firmware limiter remains active on legacy variants.

The two full-contact variants disable that limiter, a local deviation from Pollen's XL330 default.
"""


def _deployment_cfg(cfg: ArticulationCfg) -> ArticulationCfg:
    """Clone a shared geometry configuration with the audited recipe's explicit native BAM limits."""
    cfg = clone_cfg(cfg)
    servo = cfg.actuators["servos"]
    servo.actuator_effort_limit = MICRODUCK_SERVO_EFFORT_LIMIT
    servo.viscous_friction = MICRODUCK_JOINT_DAMPING
    if servo.backlash_joint_template and servo.joint_effort_limit is None:
        servo.joint_effort_limit = 0.96
    return cfg


MICRODUCK_CFG = _deployment_cfg(robots.MICRODUCK_CFG)
MICRODUCK_ALLCOLLISIONS_CFG = _deployment_cfg(robots.MICRODUCK_ALLCOLLISIONS_CFG)
MICRODUCK_ROLLERS_CFG = _deployment_cfg(robots.MICRODUCK_ROLLERS_CFG)
MICRODUCK_FULLCONTACT_CFG = _deployment_cfg(robots.MICRODUCK_FULLCONTACT_CFG)
MICRODUCK_FULLCONTACT_BACKLASH_CFG = _deployment_cfg(robots.MICRODUCK_FULLCONTACT_BACKLASH_CFG)
MICRODUCK_WALK_BACKLASH_CFG = _deployment_cfg(robots.MICRODUCK_BACKLASH_CFG)
MICRODUCK_GROUNDCONTACT_BACKLASH_CFG = _deployment_cfg(robots.MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG)
MICRODUCK_ROLLERS_BACKLASH_CFG = _deployment_cfg(robots.MICRODUCK_ROLLERS_BACKLASH_CFG)
MICRODUCK_BACKLASH_CFG = MICRODUCK_WALK_BACKLASH_CFG
MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG = MICRODUCK_GROUNDCONTACT_BACKLASH_CFG

MICRODUCK_USD_PATH = MICRODUCK_CFG.spawn.usd_path
MICRODUCK_ALLCOLLISIONS_USD_PATH = MICRODUCK_ALLCOLLISIONS_CFG.spawn.usd_path
MICRODUCK_ROLLERS_USD_PATH = MICRODUCK_ROLLERS_CFG.spawn.usd_path
MICRODUCK_FULLCONTACT_USD_PATH = MICRODUCK_FULLCONTACT_CFG.spawn.usd_path
MICRODUCK_FULLCONTACT_BACKLASH_USD_PATH = MICRODUCK_FULLCONTACT_BACKLASH_CFG.spawn.usd_path
MICRODUCK_WALK_BACKLASH_USD_PATH = MICRODUCK_WALK_BACKLASH_CFG.spawn.usd_path
MICRODUCK_GROUNDCONTACT_BACKLASH_USD_PATH = MICRODUCK_GROUNDCONTACT_BACKLASH_CFG.spawn.usd_path
MICRODUCK_ROLLERS_BACKLASH_USD_PATH = MICRODUCK_ROLLERS_BACKLASH_CFG.spawn.usd_path


MICRODUCK_FULLCONTACT_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?"
    "(fullcontact_[^/]+|[^/]+_servo_collision|(left|right)_foot_collision)/[^/]+"
)
"""The 69 active full-contact collider meshes, excluding visuals, sites, and the disabled power-support shell."""

MICRODUCK_FULLCONTACT_SERVO_SHAPE_EXPR = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?[^/]+_servo_collision/xl330"
"""Fifteen XL330 housing collision meshes (including the unactuated mouth housing), with upstream's geom names."""

MICRODUCK_FULLCONTACT_HEAD_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/"
    "(neck(/neck_pitch_1(/yaw_roll_motion(/jaw_soft)?)?)?|bearing_roll)/"
    "(fullcontact_[^/]+|[^/]+_servo_collision)/[^/]+"
)
"""Upstream head-sensor body set, including its historical right-hip bearing_roll entry."""

MICRODUCK_FULLCONTACT_TRUNK_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(fullcontact_[^/]+|[^/]+_servo_collision)/[^/]+"
)
"""Trunk-base collision meshes, excluding the disabled power-support shell."""

MICRODUCK_FULLCONTACT_BODY_NAMES = {
    "trunk": "trunk_base",
    "neck": "neck",
    "neck_pitch": "neck_pitch_1",
    "head_yaw": "yaw_roll_motion",
    "head": "jaw_soft",
    "left_foot": "ankle_left",
    "right_foot": "ankle_right",
    "left_hip_yaw": "yaw2roll",
    "right_hip_yaw": "bearing_roll",
    "left_thigh": "upper_leg_left",
    "right_thigh": "upper_leg_right",
    "left_hip": "hip_l",
    "right_hip": "hip_l_2",
    "left_shin": "leg",
    "right_shin": "leg_2",
}
"""Semantic names of physical links in both full-contact variants; rotor proxies have no colliders."""


MICRODUCK_FULLCONTACT_FOOT_SITE_OFFSETS = {
    "left": {"body": "ankle_left", "pos": (0.0, -0.0238146, -0.0140852), "rot": (0.0, -0.707107, -0.707107, 0.0)},
    "right": {"body": "ankle_right", "pos": (0.0, -0.0238146, -0.0140852), "rot": (-0.707107, 0.0, 0.0, 0.707107)},
}
"""Pinned full-contact foot sites in ankle-body frames, using Isaac Lab's xyzw quaternion convention."""


##
# Ball prop.
#
# Upstream's ball-kick task adds one free body to the scene, described by a 15-line MJCF
# (``robot/microduck/ball.xml``) holding a single analytic sphere. There is nothing for the mesh
# importer to carry, so it is authored here rather than converted -- which also means it needs no
# robot USD and is available before any robot asset has been downloaded or copied.
##

MICRODUCK_BALL_RADIUS = 0.035
"""Radius [m] of the ball prop: a 70 mm-diameter floorball, as upstream's MJCF comment describes."""

MICRODUCK_BALL_MASS = 0.015
"""Mass [kg] of the ball prop, 2.0 % of the 0.737 kg robot."""

MICRODUCK_BALL_SLIDING_FRICTION = 0.5
"""Sliding friction coefficient the ball's MJCF geom authors.

It is the one contact coefficient upstream customizes; the torsional and rolling ones are left at
MuJoCo's defaults of 0.005 and 1e-4, which are also Newton's shape defaults, so they are not
restated here.

**This coefficient is masked in every contact the ball actually makes**, on both stacks, and that is
upstream's behaviour rather than a port artefact. MuJoCo -- and Newton's MuJoCo Warp solver, which
reproduces the rule -- mixes contact friction as the element-wise *maximum* of the two shapes unless
one carries a higher ``priority``. The ground plane, the robot's shells and the two soles all sit at
1.0, so every ball contact resolves to 1.0 and this 0.5 never binds. It is carried anyway: it is
what the MJCF authors, and a surface slipperier than the ball would use it.
"""


@clone
def _spawn_microduck_ball(
    prim_path: str,
    cfg: sim_utils.SphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the ball prop and give it the inertia of a hollow shell rather than of a solid sphere.

    Upstream's MJCF states the ball's inertia outright -- ``diaginertia="1.225e-5 ..."``, which is
    exactly ``(2/3) m r^2``, a thin spherical shell. Nothing derives that from the geometry: a
    uniform-density sphere of the same mass and radius has ``(2/5) m r^2 = 7.35e-6``, 40 % less, and
    that is what a sphere prim carrying only a mass resolves to. A ball 40 % easier to spin up rolls
    and slides differently off the same kick, so the shell tensor is authored here.

    The tensor is derived from the configuration's own mass and radius rather than restated as a
    constant, so the three numbers cannot drift apart.

    Args:
        prim_path: Prim path or pattern to spawn the ball at.
        cfg: Sphere spawner configuration. Its ``mass_props.mass`` is required.
        translation: Translation w.r.t. the parent prim. Defaults to the origin.
        orientation: Orientation as (x, y, z, w) w.r.t. the parent prim. Defaults to identity.
        **kwargs: Forwarded to :func:`~isaaclab.sim.spawn_sphere`.

    Returns:
        The spawned prim.

    Raises:
        ValueError: If the configuration carries no explicit mass to derive the inertia from.
    """
    mass_cfg = next((fragment for fragment in cfg.mass_props or [] if isinstance(fragment, sim_utils.MassCfg)), None)
    if mass_cfg is None or mass_cfg.mass is None:
        raise ValueError(
            "The MicroDuck ball spawns with an explicit hollow-shell inertia derived from its mass,"
            f" so 'mass_props.mass' must be set. Received mass_props={cfg.mass_props}."
        )
    # ``spawn_sphere`` is itself clone-decorated, but this wrapper has already resolved the pattern
    # down to a single path, so the inner decorator is inert and the inertia below is authored on
    # the prototype prim the outer one copies into every environment.
    prim = sim_utils.spawn_sphere(prim_path, cfg, translation, orientation, **kwargs)
    inertia = 2.0 / 3.0 * mass_cfg.mass * cfg.radius**2
    UsdPhysics.MassAPI.Apply(prim).CreateDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
    return prim


MICRODUCK_BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        func=_spawn_microduck_ball,
        radius=MICRODUCK_BALL_RADIUS,
        # solver-common schemas rather than the PhysX ones: the ball has no backend-specific
        # property to set, and this task runs on MJWarp
        rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg()],
        collision_props=[sim_utils.UsdPhysicsCollisionCfg()],
        mass_props=[sim_utils.MassCfg(mass=MICRODUCK_BALL_MASS)],
        physics_material=UsdPhysicsRigidBodyMaterialCfg(
            static_friction=MICRODUCK_BALL_SLIDING_FRICTION,
            dynamic_friction=MICRODUCK_BALL_SLIDING_FRICTION,
            # MuJoCo has no restitution coefficient -- its bounce comes out of the contact solver
            # reference, which the MJCF leaves at the default. Zero is the matching UsdPhysics value.
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, MICRODUCK_BALL_RADIUS)),
)
"""Configuration for the ball prop of upstream's MicroDuck ball-kick task.

A free-floating, non-articulated 70 mm / 15 g hollow plastic sphere -- a floorball -- resting on the
ground 0.3 m in front of the robot. That initial position only decides where it sits before the
first reset: the ball-kick task places it in front of the kicking foot every episode, in the robot's
own yaw frame.

The MJCF gives the geom no collision masks, so it collides with everything, and upstream applies
none of the robot's collision editing to it. The mass, the radius and the sliding friction are the
MJCF's; the inertia is a hollow shell rather than the solid sphere the geometry would imply, and is
authored by :func:`_spawn_microduck_ball`.
"""
