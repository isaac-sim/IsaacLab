# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical original source extraction``, the extraction of the pinned upstream checkouts.

This is the terrain-generator variant of the task; :mod:`.flat_env_cfg` derives the flat one.
Only the rough terrain itself is unvalidated: it is registered and generated, but the foot-height
rewards and observations measure heights against the environment origin rather than against the
ground under each foot, which is exact on flat terrain only.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
import isaaclab_tasks.core.velocity.mdp as velocity_mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_CFG
from isaaclab_tasks.utils import PresetCfg

##
# Shared constants
##

MICRODUCK_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 14 servo joints in upstream MJCF order: left leg, then neck/head, then right leg.

Reference section 6b. This order is a deploy contract, not a convenience: the policy trained here
is run on the robot against a flat observation vector whose joint blocks follow the MJCF actuator
indices, and Isaac Lab resolves joints in USD order, which differs. Every joint-block observation
and the action term therefore pin it with ``preserve_order=True``.

Spelling the joints out also reproduces upstream's ``^(?!passive_).*`` selector: name resolution
matches in full, so an exact name can never pick up a ``passive_`` joint. The walk model has none
(reference section 6b), but the roller and backlash variants of the robot do.
"""

MICRODUCK_LEG_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 10 leg joints, which are the ones the posture reward holds at the stand pose.

Upstream selects them with the negative lookahead ``^(?!passive_|.*neck.*|.*head.*).*``
(reference section 2.4); spelling them out makes the selection auditable and pins their order.
"""

MICRODUCK_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 head servos, in the order the head-pose command indexes its columns (reference section 6).

This order is a contract between the command term and the two head rewards, not a preference, so
every selection of these joints pins it with ``preserve_order=True``.
"""

MICRODUCK_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in upstream's ``[left, right]`` site order (reference section 2.9)."""

MICRODUCK_TRUNK_BODY_NAME = "trunk_base"
"""Body upstream measures the base pose, upright reward and self-collisions on."""

MICRODUCK_HEAD_BODY_NAMES = ["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"]
"""Body-name patterns the head centre-of-mass randomization perturbs.

The MJCF names the second body ``neck_pitch``, which collides with the joint of the same name, so
the conversion disambiguates it -- currently to ``neck_pitch_1``. The optional numeric suffix
matches either spelling, and a pattern that matches nothing raises, so replacing the asset
cannot silently drop a body from the selection.

Upstream's list (reference section 2.6) is
``("neck", "neck_pitch", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)", "bearing_roll")``, and
this port differs from it in two ways, both deliberate:

* ``bearing_roll`` is **dropped**. It is the right hip-yaw link, not a head body; upstream's own
  comment says it "has always been listed here by mistake and is kept only to preserve existing DR
  behavior". Carrying a documented mistake forward would randomize the right leg's mass
  distribution under the name of the head, on a schedule tuned for the head.
* ``bottom_head_shell`` is dropped because the walk model has no such body -- the alternation
  ``(bottom_head_shell|jaw_soft)`` resolves to ``jaw_soft`` alone on ``robot_walk.xml``. It exists
  only in the all-collisions variants, which this task does not use.
"""

MICRODUCK_ALLCOLLISIONS_COLLIDER_XFORMS = [
    "left_foot_collision",
    "right_foot_collision",
    "leg_1",
    "hip_l_1",
    "top_head_shell_1",
    "jaw_1",
    "bottom_head_shell_1",
    "np_f970_1",
]
"""Xforms holding the ten enabled colliders of the all-collisions model, by disambiguated name.

The MJCF names only the two soles; the prebuilt USD names the rest after their meshes.
The importer appends a ``_1`` suffix wherever that name is
already taken by the visual mesh. Four of these names cover two colliders each -- ``leg_1`` and
``hip_l_1`` appear once per leg, and the three head entries all hang off ``jaw_soft`` -- which is why
the list has eight entries for ten colliders.

This is the all-collisions model's world-contact set, which on that model is also its self-collision
set: the asset disables every other collider, and the only geometry it leaves out of world
contact, the trunk's ``power_support`` shell, is disabled outright because the importer cannot
represent the MJCF's ``contype``/``conaffinity`` masks.
"""

MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(" + "|".join(MICRODUCK_ALLCOLLISIONS_COLLIDER_XFORMS) + ")/[^/]*"
)
"""Shape-level prim expression selecting those ten colliders, for the Newton contact sensor.

Shape expressions full-match against shape paths rather than body paths, so the trailing ``/[^/]*``
is what reaches the collider prim below its Xform, and the optional ``(.*/)?`` spans the bodies
between the trunk and each collider -- including none at all, for the battery pack bolted to the
trunk itself.

Used as *both* the sensing and the filter expression, this is the many-to-many self-collision sensor
upstream has: every enabled collider against every other one, rather than the trunk against a list
of bodies. It is a Newton-backend capability
(:attr:`~isaaclab.sensors.ContactSensorCfg.sensor_shape_prim_expr` and
:attr:`~isaaclab.sensors.ContactSensorCfg.filter_shape_prim_expr`); the body-level
:attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` resolves a force matrix only for a
``prim_path`` matching one prim per environment, which is what forced the narrower sensor before.
"""

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

##
# Sensor imperfections the actor is trained against
##

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle (reference section 6).

The angle is drawn once per robot and never resampled, so this bounds a fixed mounting error rather
than a disturbance. Both misaligned terms read the *same* rotation, so they must be configured with
this same bound -- the shared accessor raises rather than serve two different ones.
"""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency (reference section 8).

The lag itself is 0 or 1 step. Re-drawing it every step would give a flicker the policy can filter
out; holding it for tens of steps makes it a latency regime the policy has to tolerate instead.
"""

##
# Foot-height targets
##

MICRODUCK_SOLE_TO_ANKLE_OFFSET = 0.0225
"""Height [m] of the ankle body frame above the sole, with the foot level.

Upstream measures foot height at the ``left_foot`` / ``right_foot`` *sites*, which sit on the sole,
using a terrain height sensor this port has no equivalent of; the port measures the **ankle body
frame** instead (see ``rewards._feet_height_above_ground``). Measured on the pinned
``robot_walk.xml`` with MuJoCo forward kinematics at the STAND2 home pose: the ankle body origin is
0.022496 m above the foot site, and the site is 0.000054 m above the lowest vertex of the sole
collision mesh, so the sole and the site are the same surface to within 0.05 mm.
"""

MICRODUCK_FOOT_TARGET_HEIGHT = 0.02 + MICRODUCK_SOLE_TO_ANKLE_OFFSET
"""Swing-height target [m] in the ankle-body frame, 0.0425 m.

Upstream's target is 0.02 m at the sole (reference section 2.4). Copying that number onto an
ankle-frame measurement would ask the robot to bury its feet 2 cm into the ground, so the offset
above is added instead.
"""

_FOOT_SWING_HEIGHT_WEIGHT = -0.25 * (MICRODUCK_FOOT_TARGET_HEIGHT / 0.02) ** 2
"""Swing-height penalty weight, -1.12890625.

``foot_swing_height`` charges a *relative* error, ``(peak / target - 1)^2``, which -- unlike
``foot_clearance``'s absolute ``|h - target|`` -- is not invariant under moving the measurement
frame. Substituting ``peak = peak_sole + offset`` and ``target = 0.02 + offset`` gives
``((peak_sole - 0.02) / target)^2``, i.e. upstream's ``((peak_sole - 0.02) / 0.02)^2`` scaled by
``(0.02 / target)^2``. Scaling the weight by the reciprocal makes the term numerically identical
to upstream's rather than 4.5x weaker.
"""

##
# Curriculum stage tables (reference section 6). Steps are in environment steps: upstream schedules
# by PPO iteration and collects ``num_steps_per_env`` of them per iteration.
##

MICRODUCK_STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, matching ``MicroDuckPPORunnerCfg.num_steps_per_env``.

Imported by ``MicroDuckPPORunnerCfg`` so the two cannot drift apart; a runner change here would
otherwise silently desynchronize every curriculum boundary from "iteration N".
"""


def _iterations(count: int) -> int:
    """Convert an upstream iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Terrain
##

MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    # 1 mm height quantization, so a 1.7 deg slope is a smooth ramp rather than a staircase of
    # 5 mm ledges the robot cannot step over. Upstream sets this on the slope sub-terrain; Isaac
    # Lab's generator pushes its own value down onto every height-field sub-terrain, and the slope
    # is the only one, so it is set here instead.
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        # MicroDuck lifts its feet 1-2 cm, so steps are capped at 1.5 cm rather than the 23 cm of
        # the stock ROUGH_TERRAINS_CFG
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # uneven cobblestone-like ground. The 0.45 m cell is upstream's: a 0.12 m cell would put
        # 66x66 boxes on each 8 m patch, which does not fit in memory at 200 patches.
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),
            platform_width=1.5,
        ),
        # gentle slopes, rise over run: 0.03 to 0.10 is 1.7 to 5.7 degrees by difficulty. The
        # platform is on top and the pyramid is not inverted, so a reset never places the robot at
        # the bottom of a pit -- upstream dropped both inverted variants for exactly that reason.
        # ``border_width`` is left at 0.0, as upstream leaves it. The stock ROUGH_TERRAINS_CFG rings
        # its slopes with a 0.25 m flat border, but a pyramid slope is already level with its
        # neighbours at the patch edge -- the height field is ``height_max * xx * yy`` with
        # ``xx = yy = 0`` there -- so a border would only shorten the slope for no continuity gain.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
        ),
    },
)
"""Rough terrains MicroDuck trains on (reference section 6, ``MICRODUCK_ROUGH_TERRAINS_CFG``).

Far gentler than :data:`~isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`, which is sized for a
metre-tall quadruped. Upstream's box sub-terrain classes map onto Isaac Lab's ``Mesh*`` ones and its
height-field slope onto ``Hf*``; the proportions, sizes and ranges are upstream's.

Upstream additionally softens the terrain contacts (``solref`` 0.02 -> 0.04 s) to damp the impulsive
forces a foot landing on a box edge produces. Isaac Lab has no per-terrain-geometry contact
stiffness override, so that is not ported; it is the first thing to try if the rough task shows
contact-force spikes.
"""


##
# Physics presets
##


@configclass
class MicroDuckPhysicsCfg(PresetCfg):
    """Backend presets for the MicroDuck velocity environments.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend. The solver limits
    are upstream's rough-terrain profile (reference section 1): ``njmax`` 1500, ``nconmax`` 200,
    30 solver iterations and 50 line-search iterations. The flat variant tightens them.

    The environment requires Newton-native BAM drives and MJWarp's friction-constraint bridge.
    Other physics backends are not offered by this task.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1500,
            nconmax=200,
            iterations=30,
            ls_iterations=50,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # upstream steps MuJoCo once per 0.005 s physics tick, so the substep count stays at one
        # and ``sim.dt`` is the upstream timestep (reference section 1)
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckSceneCfg(InteractiveSceneCfg):
    """Scene with the MicroDuck robot on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MICRODUCK_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the robot walks on is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Feet and trunk only: upstream tracks ground contact on the two soles and self-collision on
    # the trunk subtree (reference section 2.9). No height scanner -- upstream drops the terrain
    # scan and the matching observation for MicroDuck (reference sections 2.3 and 2.9).
    #
    # The pattern is deliberately unlike the flat ``Robot/[^/]*`` other locomotion tasks use. The
    # MJCF import nests every link under the trunk -- the ankle links the soles hang off sit four
    # levels below it -- so a single-level pattern matches nothing here. Newton compiles this as one
    # plain regular expression and full-matches it against body labels, which are whole prim paths,
    # so ``.*`` crosses ``/`` and reaches them. That is a Newton-only property: the PhysX backend
    # matches prim paths one path token at a time and would need a different expression, which is
    # consistent as long as :class:`MicroDuckPhysicsCfg` offers only the MJWarp backend.
    #
    # Bodies resolve in label order, ``[trunk_base, ankle_right, ankle_left]``, which is neither the
    # order below nor upstream's ``[left, right]``: select feet by name with ``preserve_order=True``.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # Self-collision sensor. Upstream filters the trunk subtree against itself and counts the
    # contact slots that report found (reference section 2.9); ``contact_forces`` above is
    # unfiltered and produces no force matrix at all, so ``self_collision_cost`` needs its own
    # sensor.
    #
    # Sole against sole is the whole self-collision signal *on the converted asset*, which is
    # narrower than upstream's. The walk model has five collidable geometries: the two soles, and
    # ``power_support`` (the battery holder on the trunk) plus the ``leg`` mesh on each shin, all
    # three in the MJCF's ``self_collision_only`` class (``contype=2 conaffinity=2``) so they
    # collide with each other but never with the ground. Upstream's subtree sensor therefore also
    # watched the shins hitting the battery holder and each other, which is the channel that class
    # exists for.
    #
    # The MJCF-to-USD importer cannot represent ``contype`` / ``conaffinity`` masks, so those three
    # geometries would otherwise arrive as ordinary *world* colliders that stub on the ground.
    # The prebuilt asset disables them to preserve the MJCF's world-contact set. They have no
    # collision role left, so this sensor cannot see them.
    #
    # LOST GUARD-RAIL: shin-versus-battery-holder and shin-versus-shin contacts are unpenalized
    # here. This is a *collider* gap, not a sensing one: those geometries have no collision role on
    # this model, so no sensor can see them. If a future asset restores their self-collision role,
    # widen this sensor the way the stand-up and roulade tasks are widened --
    # ``sensor_shape_prim_expr``/``filter_shape_prim_expr`` over
    # :data:`MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR`, with ``saturate=True`` on the reward --
    # rather than by growing this body-level pair.
    #
    # Until then this sensor stays exactly as it is. Its one-against-one shape is not a workaround
    # for a missing capability, it is the whole reachable signal on the walking model, and its
    # recipe is validated: one sensing body against one filter body reports 0 or 1 as upstream's
    # single contact slot does, where sensing both would count the one contact twice, once from
    # each side.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_right"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Reference section 2.7. The resampling range, heading controller, heading range and the
    # forward-only fraction are inherited from upstream's base template; the velocity ranges, the
    # standing fraction and the turn-in-place fraction are MicroDuck's own. ``rel_heading_envs =
    # 0.0`` leaves the heading controller configured but inert, which is what upstream ships.
    #
    # The two bucket fractions are what ``MicroDuckVelocityCommand`` adds over the stock term: a
    # fifth of the environments walk straight forward and 15% turn on the spot. The buckets
    # overlap, and the term reproduces upstream's precedence.
    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.15,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    # Reference section 6. A joint-position delta from the stand pose for
    # ``(neck_pitch, head_pitch, head_yaw, head_roll)``; the ranges start almost closed and the
    # ``head_pose_range`` curriculum opens them. Looking around on command is a primary MicroDuck
    # objective, not a disturbance: ``head_pose_tracking`` carries weight 2.0, the same as velocity
    # tracking.
    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )

    # Reference section 6. A trunk-pose delta, ``(x, y, z)`` [m] then ``(roll, pitch, yaw)`` [rad].
    # Upstream keeps this command alive at tiny ranges with its reward weight at 0.0, so that the
    # observation block the deployed policy expects stays populated and the term can be switched on
    # without changing the network shape.
    body_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.05, 0.05),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Reference sections 3 and 9: the target is ``default_joint_pos + action * scale``, and
    # MicroDuck raises the base template's 0.5 scale to 1.0, so an action is a joint-space offset
    # from the stand pose in radians.
    #
    # The biased variant additionally subtracts the encoder bias the actor's ``joint_pos``
    # observation adds, which is the loop a real servo closes on its own miscalibrated encoder: the
    # policy commands ``a``, reads ``a`` back, and the joint actually settles at
    # ``default + a - bias``. Wiring only one half of that loop would train a permanent joint-space
    # offset into the policy instead of a calibration error it has to tolerate.
    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    The actor reads the robot the way the deployed runtime does -- through a miscalibrated encoder,
    a misaligned IMU, a bus latency and per-step noise -- while the critic reads the true state plus
    the privileged quantities the robot has no sensor for.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        Term order **is** the deployed observation layout (reference section 7), which the runtime
        on the robot rebuilds by hand from its own sensor reads: base angular velocity (3),
        projected gravity (3), joint positions (14), joint velocities (14), the previous action
        (14), then the command block ``[twist(3), head_pose(4), body_pose(6)]``. Moving a term or
        changing its width silently invalidates every trained checkpoint, so the layout is pinned by
        a test rather than left to term declaration order alone.

        Noise magnitudes are MicroDuck's own overrides of the base template (reference section 2.3),
        an order of magnitude tighter throughout. On top of the noise the actor carries the three
        systematic corruptions per-step noise cannot express (reference sections 2.3, 6 and 8):

        * a constant per-robot **encoder bias** on ``joint_pos``, closed against
          :class:`~isaaclab_tasks.contrib.microduck.pollen_mdp.actions.BiasedJointPositionAction`;
        * a constant per-robot **IMU mounting misalignment** on the two IMU terms, which share one
          rotation and therefore must agree on ``max_angle_deg``;
        * a stochastic **bus latency**: 0 or 1 control step on the IMU, re-drawn at most every 64
          steps, and a constant 1 step on ``joint_vel``, whose firmware derives velocity from the
          previous position-sample window.

        Isaac Lab applies the noise to the delayed value, where mjlab delays the already-noised one.
        Both give a stale signal plus an independent uniform draw, so the observation distribution
        is the same; see
        :class:`~isaaclab_tasks.contrib.microduck.pollen_mdp.observations.delayed_observation`.
        """

        base_ang_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.base_ang_vel_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.projected_gravity_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_biased,
            params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.joint_vel_rel,
                "term_params": {"asset_cfg": _SERVO_JOINT_CFG},
                "min_lag": 1,
                "max_lag": 1,
                "update_period": 0,
            },
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (reference sections 2.2 and 2.3).

        Upstream's critic is the actor's term dictionary with every corruption removed, plus what
        the robot has no sensor for: the base linear velocity -- which MicroDuck deletes from the
        actor because the deployed runtime cannot measure it -- and four foot terms read off the
        contact sensor. The command block is appended to both groups, so it ends this one too.

        ``enable_corruption=False`` strips the noise, but it gates neither the delay nor the
        misalignment, so the terms here are the stock, undelayed, unmisaligned ones rather than
        references to the actor's. This group is not a deploy contract, so its width is measured
        rather than declared.

        ``foot_height`` measures the body height above the environment origin, which is the
        clearance over the ground on flat terrain only; on the rough task the critic's foot heights
        follow the trunk over the terrain instead of the terrain under the foot.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_height = ObsTerm(func=mdp.foot_height_safe, params={"asset_cfg": _FOOT_BODY_CFG})
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events.

    Reference section 2.6. Upstream's ``base_com`` term is deliberately absent: it selects zero
    bodies upstream and MicroDuck never fills it in, so it is a documented no-op there (reference
    section 2.6a). Its ``expand_bam_friction_fields`` term has no counterpart because it only
    registers MuJoCo's ``dof_frictionloss``/``dof_damping`` for per-world expansion, which the BAM
    actuator's per-environment storage already is; its ``reset_action_history`` has none because
    Isaac Lab's action manager resets its own buffers.

    The three randomizations upstream ships disabled -- motor gains, joint damping and base
    orientation -- are not carried over at all, since a term that is never enabled is not part of
    the recipe (reference section 2.6).
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

    # Upstream sets the sole geometries' friction outright (``operation="abs"``). On Newton this
    # term samples a single friction coefficient per shape, which is exactly MuJoCo's model, and
    # ignores ``dynamic_friction_range`` and ``num_buckets``; both are supplied because the
    # signature requires them and the PhysX backend would need them. The soles are the only
    # collidable geometries on the robot, and the ground plane combines friction multiplicatively
    # at 1.0, so this range *is* the friction the robot walks on.
    #
    # Upstream draws one value shared by both feet (``shared_random=True``); this term draws one per
    # foot, which is a superset -- it also covers a robot with one worn sole.
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _FOOT_MATERIAL_CFG,
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Startup, not reset: a calibration error is a property of the robot, and resampling it every
    # episode would let the policy average it out instead of learning to tolerate it.
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

    # +/-5 % on the trunk. Upstream perturbs the trunk's pseudo-inertia with a single log-Cholesky
    # scale ``alpha`` drawn from ``(log(0.95)/2, log(1.05)/2)``, i.e. one sample scaling mass and
    # inertia together. ``recompute_inertia=True`` reproduces that coupling: the inertia is scaled
    # by the same mass ratio. What is *not* reproduced is upstream's perturbation of the inertia
    # *shape* -- the ratios between the principal axes stay nominal here.
    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    ##
    # Reset: redrawn every episode.
    ##

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Upstream samples the absolute base height in (0.12, 0.13) m; Isaac Lab samples an
            # offset from the configured default, which is the 0.125 m midpoint of that band.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # Trunk centre of mass, +/-3 mm on every axis to start with; the ``com_range`` curriculum opens
    # it to +/-15 mm. Isaac Lab's term takes a per-axis dictionary where upstream takes one
    # symmetric range for all three, which is what the curriculum writes back.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    # The same, on the head bodies, ramped to +/-10 mm. A head that is heavier on one side than the
    # model says is what a real battery or camera mount produces, and the head is the mass the
    # walking controller is least able to compensate for.
    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    # +/-10 % on the BAM servos' gearbox friction budget. Upstream names the term
    # ``randomize_joint_friction``, but it does not touch the solver's joint friction: it scales the
    # actuator's own Coulomb, Stribeck and load-dependent budget, which is what a worn or freshly
    # greased gearbox changes. The stock ``randomize_actuator_gains`` cannot express it -- it writes
    # PD gains this model never reads.
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    # +/-10 % on every joint's armature: the reflected rotor inertia of a servo is a fitted quantity,
    # not a measured one. Upstream redraws it every reset and this port follows, even though the
    # term writes through host tensors, so that the schedule matches.
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    ##
    # Interval.
    ##

    # MicroDuck replaces the base template's much larger six-axis shove with a planar nudge every
    # 3 to 6 s: a 0.74 kg robot that is hit at 0.5 m/s in yaw is not recoverable, so upstream would
    # only be training the fall.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Upstream's full recipe (reference sections 2.4 and 5 for the base template's terms, section 6
    for MicroDuck's own). Every weight and parameter here is upstream's except the two foot-height
    targets and the swing-height weight, which are re-derived in
    :data:`MICRODUCK_FOOT_TARGET_HEIGHT` because this port measures foot height at the ankle body
    rather than at the sole site.

    Upstream's ``soft_landing`` term is absent because upstream removes it from its own recipe, and
    there is no energy or torque penalty because the mjlab base template has none.
    """

    ##
    # Velocity tracking (reference section 5).
    ##

    track_lin_vel = RewTerm(
        func=mdp.track_linear_velocity,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )

    ##
    # Posture (reference section 5).
    ##

    # A bounded Gaussian *reward* on the trunk's gravity tilt, not the stock unbounded L2 penalty.
    upright = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )

    # Hold the legs near the stand pose, with a tolerance that widens once the robot is asked to
    # walk. The neck and head joints are excluded: they follow the head-pose command instead.
    # ``std_running`` repeats ``std_walking`` because upstream sets them equal, which leaves the
    # running regime inactive at MicroDuck's 0.4 m/s command ceiling.
    pose = RewTerm(
        func=mdp.pose_mode_switch,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {
                ".*hip_yaw": 0.1,
                ".*hip_roll": 0.05,
                ".*hip_pitch": 0.15,
                ".*knee": 0.15,
                ".*ankle": 0.1,
            },
            "std_walking": {
                ".*hip_yaw": 0.3,
                ".*hip_roll": 0.05,
                ".*hip_pitch": 0.4,
                ".*knee": 0.4,
                ".*ankle": 0.25,
            },
            "std_running": {
                ".*hip_yaw": 0.3,
                ".*hip_roll": 0.05,
                ".*hip_pitch": 0.4,
                ".*knee": 0.4,
                ".*ankle": 0.25,
            },
            "walking_threshold": 0.01,
            "running_threshold": 1.5,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    ##
    # Trunk regularizers (reference section 5).
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # The ``action_rate_weight`` curriculum ramps this to -1.0 by iteration 1500.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    ##
    # Gait and feet (reference section 5).
    ##

    air_time = RewTerm(
        func=mdp.feet_air_time_windowed,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=-2.0,
        params={
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.foot_swing_height,
        weight=_FOOT_SWING_HEIGHT_WEIGHT,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "asset_cfg": _FOOT_BODY_CFG,
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "command_threshold": 0.01,
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG},
    )

    ##
    # Head and body pose tracking (reference section 6).
    ##

    head_pose_tracking = RewTerm(
        func=mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
    )
    # Weight 0.0, ramped to +3.0 from iteration 600 by the ``head_pose_bias_weight`` curriculum.
    # The weight is **positive** because the term already returns a non-positive value.
    head_pose_bias = RewTerm(
        func=mdp.head_pose_bias_penalty,
        weight=0.0,
        params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": _HEAD_JOINT_CFG},
    )
    # Weight 0.0 throughout: upstream keeps the slot alive but never trains against it.
    body_pose_tracking = RewTerm(
        func=mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Reference section 2.5. Upstream terminates on tilt rather than on trunk contact, which suits a
    robot whose trunk is its only substantial body.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # Inert on the flat task, which is a plane: the term short-circuits to all-false there, exactly
    # as upstream's ``out_of_terrain_bounds`` does.
    #
    # The buffer reproduces upstream's bound. Upstream trips at
    # ``|x| > num_rows * size_x / 2 - 0.3``; this term trips at ``|x| > map_width / 2 - buffer``
    # with ``map_width = num_rows * size_x + 2 * border_width``, so the same bound needs
    # ``buffer = border_width + 0.3 = 20.3``. The default 3.0 would let the robot walk 17 m into the
    # border before terminating.
    out_of_terrain_bounds = DoneTerm(
        func=velocity_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"distance_buffer": 20.3},
    )

    # Reference section 6. Catches a *broken* robot rather than a fallen one: a NaN state compares
    # false against every bound, so no other termination would fire on it and the environment would
    # train on garbage forever.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Reference sections 2.8 and 6. Upstream deletes the base template's ``command_vel`` curriculum
    outright -- MicroDuck's velocity ranges are fixed -- and keeps ``terrain_levels`` only on rough
    terrain. Everything else here is MicroDuck's own.

    The schedules are step functions of the global environment-step count. Upstream writes them in
    PPO iterations, which :func:`_iterations` converts.
    """

    # Only meaningful with a terrain generator; :class:`.flat_env_cfg.MicroDuckVelocityFlatEnvCfg`
    # removes it, as upstream does.
    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.1},
                {"step": _iterations(500), "weight": -0.2},
                {"step": _iterations(750), "weight": -0.4},
                {"step": _iterations(1000), "weight": -0.6},
                {"step": _iterations(1250), "weight": -0.8},
                {"step": _iterations(1500), "weight": -1.0},
            ],
        },
    )

    # Held at 0.0 until the robot walks: an early bias penalty rewards holding the head still, which
    # is the opposite of what the head-pose command asks for.
    head_pose_bias_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(600), "weight": 1.0},
                {"step": _iterations(1000), "weight": 2.0},
                {"step": _iterations(1500), "weight": 3.0},
            ],
        },
    )

    standing_envs = CurrTerm(
        func=mdp.standing_envs_stages,
        params={
            "command_name": "base_velocity",
            "standing_stages": [
                {"step": _iterations(0), "rel_standing_envs": 0.02},
                {"step": _iterations(500), "rel_standing_envs": 0.05},
                {"step": _iterations(750), "rel_standing_envs": 0.1},
                {"step": _iterations(1000), "rel_standing_envs": 0.15},
                {"step": _iterations(1500), "rel_standing_envs": 0.2},
                {"step": _iterations(2000), "rel_standing_envs": 0.25},
            ],
        },
    )

    # Opens the head command from a few centiradians to nearly the full servo travel.
    #
    # The final stage is upstream's verbatim. Note that the pinned ``robot_walk.xml`` limits
    # ``neck_pitch`` to +1.0472 rad, so the +/-1.10 cap over-commands it by 0.05 rad at the top of
    # the schedule and the head-pose reward can never be fully earned there. Upstream trained with
    # this mismatch, and the port keeps it rather than silently retuning a value the deployed
    # policy was trained against.
    head_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {
                    "step": _iterations(0),
                    "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
                },
                {
                    "step": _iterations(500),
                    "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047)),
                },
                {
                    "step": _iterations(1000),
                    "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11)),
                },
                {
                    "step": _iterations(1500),
                    "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20)),
                },
                {
                    "step": _iterations(2000),
                    "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31)),
                },
            ],
        },
    )

    # A single stage, so the body-pose ranges never move. Kept as a curriculum term because that is
    # where upstream declares them and where a follow-up would widen them.
    body_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {
                    "step": _iterations(0),
                    "ranges": (
                        (-0.005, 0.005),
                        (-0.005, 0.005),
                        (-0.005, 0.005),
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                    ),
                },
            ],
        },
    )

    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
                {"step": _iterations(1500), "range": 0.015},
            ],
        },
    )

    head_com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_head_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck velocity-tracking environment on rough terrain."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckPhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
    scene: MicroDuckSceneCfg = MicroDuckSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the deployed policy runs
        # at, and 20 s episodes are 1000 control steps (reference section 1).
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, which is what upstream does: the
        # controller executes inside the MJWarp step and republishes the live friction budget and
        # viscous coefficient into ``dof_frictionloss``/``dof_damping`` every physics step, instead
        # of clipping the torque Lab-side once per control step. The decimation above is even, which
        # is what lets the stateful servo delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # the terrain-level curriculum needs the generator to lay its sub-terrains out by difficulty
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
