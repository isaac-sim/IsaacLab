# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roller-skating velocity environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks2 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the shared mjlab base template live in the companion
``historical original source extraction`` and are cited as "reference section N".

The robot wears skates: :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_ROLLERS_CFG` replaces each foot with a
two-wheel bogie, so the two soles give way to four freely spinning tires and the model carries 18
hinges instead of 14 (addendum section 2.3). The four wheels are undriven -- the action space is the
same 14 servos as every other MicroDuck task -- so the robot cannot walk and can only get anywhere by
*pushing*, which is what makes this a different task rather than a different robot.

Three things follow from that and shape the whole recipe:

* **The surge command is a throttle, not a velocity target.** ``cmd_x`` is ``0`` to coast, positive
  to push and negative to brake, and no term tracks a commanded speed. The walking recipe's
  ``track_linear_velocity`` / ``track_angular_velocity`` are therefore deleted, and the single
  positive task reward is :attr:`RewardsCfg.wheel_speed`: the wheels have to actually turn.
* **The gait is a stroke, not a step.** Six terms shape it -- air time, glide, single support, gait
  symmetry, flat blades and forward lean -- and their common enemy is the *swizzle*, the degenerate
  both-blades-down waddle that a naive contact reward converges to. Upstream's comments record it
  being rediscovered repeatedly.
* **Turning is switched off.** The heading machinery is computed and then clamped to zero
  (addendum section 7.21), so the policy learns straight-line skating and
  :attr:`RewardsCfg.heading_hold` is what keeps it from veering.

Two upstream numbers are reproduced **verbatim although they are known to be stale**, because the
deployed skating policies were trained with them and this port's job is parity: the
:attr:`RewardsCfg.com_height_target` band and :attr:`RewardsCfg.wheel_speed`'s wheel radius. Both are
documented where they are configured.

The task is registered as a variant of the velocity family, which is where upstream registers it
(``Mjlab-Velocity-Flat-MicroDuck-Rollers``) and where its recipe comes from: it re-derives the mjlab
velocity template term by term rather than starting from the stand-up or roll one, and it shares this
package's joint-name, curriculum-step and PPO-runner definitions. It lives in this package for the
same reason, next to :mod:`.flat_env_cfg`, rather than in a package of its own.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ROLLERS_CFG
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# Roller-model constants (addendum section 2.3)
##

MICRODUCK_WHEEL_JOINT_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
"""The four undriven wheel hinges, in the model's left-front-to-right-rear order.

They are interleaved among the servos in the MJCF -- two after each ankle -- and the conversion
resolves joints in Newton's order, so nothing here may be selected by index. Upstream routes every
index-keyed parameter through a memoized ``find_joints(r"^(?!passive_).*")`` for the same reason.
"""

MICRODUCK_ROLLERS_JOINT_NAMES = MICRODUCK_JOINT_NAMES + MICRODUCK_WHEEL_JOINT_NAMES
"""All 18 hinges of the roller model: the 14 servos in deploy order, then the four wheels.

Only the posture reward selects this. Upstream leaves that term's base-template selector at ``.*``,
which picks up the wheels, and neutralizes them with a standard deviation of 999 instead of narrowing
the selection -- the reward is a *mean* over the selected joints, so narrowing it to 14 would change
the value of every posture score (addendum section 7.20). Reproducing the 18-joint mean is why the
wheels are named here.
"""

MICRODUCK_TIRE_BODY_NAMES = ["tire", "tire_2", "tire_3", "tire_4"]
"""The four tire bodies, ordered ``[left-front, left-rear, right-front, right-rear]``.

This order is a contract, not a preference. It is the order the contact sensor reports slots in, and
every foot term folds consecutive pairs of slots into one foot, so a permutation would silently swap
the left and right feet -- which
:class:`~isaaclab_tasks.contrib.microduck.pollen_mdp.rewards.gait_symmetry_penalty` reads directly.
"""

MICRODUCK_FOOT_BODY_NAMES = ["ankle_l_v1", "ankle_r_v1"]
"""The two ankle bodies, in upstream's ``[left, right]`` order.

The roller model renames the walking model's ``ankle_left`` / ``ankle_right`` and moves the sole
collider onto the tires hanging off them, so these bodies carry no collider at all -- they are the
*frame* the blade-flatness reward measures, not a contact body.
"""

MICRODUCK_TIRES_PER_FOOT = 2
"""Contact bodies making up one foot on this model.

Upstream gets the family's two-slot, left-first foot semantics from a ``mode="subtree"`` contact
sensor that reduces each ankle's subtree -- its two tires -- to a single slot (addendum section 5.2).
Isaac Lab's contact sensor reports one slot per body, so the reduction happens in the terms instead;
see :func:`~isaaclab_tasks.contrib.microduck.pollen_mdp.observations.fold_bodies_into_feet`.
"""

MICRODUCK_FOOT_NORMAL_AXIS = (0.0, 1.0, 0.0)
"""Sole normal of a skate blade, in the ankle body frame.

Upstream measures blade flatness at the ``left_foot`` / ``right_foot`` MJCF **sites**, which Isaac Lab
has no equivalent of. Read off the converted asset, both sites are rotated relative to their ankle
body -- 180 degrees about ``(0, 1, 1)/sqrt(2)`` on the left, -90 degrees about ``x`` on the right --
and both rotations carry the site's ``z`` axis onto the ankle body's ``+y`` axis, so measuring the
body frame about this axis reproduces upstream's quantity on both feet.

Confirmed in simulation rather than only on paper: with the robot upright at the stand pose the
gravity direction in either ankle body frame is ``(0.000, -0.996, 0.087)``, i.e. within 5 degrees of
this axis on both feet, which no other body axis is anywhere near.
"""

MICRODUCK_ROLLERS_STANDING_HEIGHT = 0.14070
"""Height [m] of ``trunk_base`` with the joints at the stand pose and the tires just touching the ground.

Measured geometrically on the pinned ``robot_allcollisions_rollers.xml`` (addendum section 2.3), with
``tire_3`` as the limiting collider, and reproduced by the converted asset to 0.03 mm: with the joints
at the stand pose the tire centres sit 0.12573 m below ``trunk_base``, and 0.12573 + 0.0150 m of tire
radius is 0.14073 m.

Under the robot's own weight the trunk rests about 2.7 mm lower, at 0.1380 m, which is the
compliance of MuJoCo's default contact constraint rather than a conversion gap -- a ``solref`` time
constant of 0.02 s admits ``g * tau^2 = 3.9 mm`` of penetration at equilibrium, and upstream's
contacts carry the same constant.

Recorded here because two of this task's numbers are sized against a *different* model and only make
sense next to it; see :data:`MICRODUCK_ROLLERS_SPAWN_HEIGHT` and :attr:`RewardsCfg.com_height_target`.
"""

MICRODUCK_ROLLERS_SPAWN_HEIGHT = 0.1385
"""Spawn height [m] of ``trunk_base``, the midpoint of upstream's reset band ``(0.1335, 0.1435)``.

:data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_ROLLERS_CFG` deliberately ships the walking model's 0.125 m
and
warns that a roller task must override it: wheels are taller than soles, and 0.125 m would bury the
tires 1.6 cm in the floor. This is that override, and it is set to upstream's band midpoint rather
than to :data:`MICRODUCK_ROLLERS_STANDING_HEIGHT` so that :attr:`EventsCfg.reset_base`'s symmetric
``+/-5 mm`` offset reproduces upstream's absolute band exactly -- Isaac Lab samples the reset height
as an offset from the configured default where upstream samples it absolutely.

The band straddles the geometric standing height, from 7.2 mm below it to 2.8 mm above, so a reset is
a millimetre-scale settle onto the tires rather than a clean placement. That is upstream's, not this
port's: it is what the deployed policies were trained from. The midpoint itself lands within 0.5 mm
of the loaded rest height of 0.1380 m.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_ROLLERS_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_HIP_ROLL_JOINT_CFG = SceneEntityCfg("robot", joint_names=["left_hip_roll", "right_hip_roll"], preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_WHEEL_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_TIRE_BODY_NAMES, preserve_order=True)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, matched to the velocity task."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency (reference section 8)."""


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckRollersPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck roller-skating environment.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend and the only one
    offered. Upstream inherits the mjlab base template's solver limits unchanged and never revisits
    them for this task (addendum section 7.4); this port measures them instead, because the roller
    model puts four wheels on the ground where the walking model puts two soles.

    MJWarp is also the only backend that can run this task as configured: the environment sets
    ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected on the PhysX
    family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Measured, not inherited: profiled under random actions at 256 and 2048 environments
            # with the tilt termination removed, so the robots sprawl and every collider reaches the
            # floor. **83 constraints and 26 contacts** per environment at 2048 environments, 74 and
            # 25 at 256 -- the contact peak agrees to one between the two scales, which is what says
            # the tail has been sampled. Logs:
            # ``artifacts/microduck/profile_microduck_contacts_rollers_{256,2048}envs.log``, from
            # ``artifacts/microduck/profile_microduck_contacts.py``.
            #
            # ``njmax`` is a hard per-environment cap and carries the wider margin; ``nconmax`` is a
            # per-environment share of one shared buffer and cannot overflow at the measured peak, so
            # it sits just above it, at the same 1.2x the stand-up and roll tasks use.
            njmax=160,
            nconmax=32,
            # upstream's flat solver profile, which this task inherits unchanged
            iterations=10,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # upstream steps MuJoCo once per 0.005 s physics tick (reference section 1)
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckRollersSceneCfg(InteractiveSceneCfg):
    """Scene with the roller-skating MicroDuck on a ground plane, and two contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the tires roll on is the one the MJCF authors on them.
            # Upstream deletes its foot-friction randomization on this task with the one-line reason
            # that *wheels roll; ground friction lives in the XML*, and this port follows.
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The roller model, spawned at a height its own wheels support. See
    # :data:`MICRODUCK_ROLLERS_SPAWN_HEIGHT` for why the asset's own 0.125 m is overridden here rather
    # than corrected in the asset.
    robot = MICRODUCK_ROLLERS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=MICRODUCK_ROLLERS_CFG.init_state.replace(pos=(0.0, 0.0, MICRODUCK_ROLLERS_SPAWN_HEIGHT)),
    )

    # Trunk and the four tires. Unlike the walking task's, this sensor cannot watch the ankle bodies:
    # on this model they carry no collider and would report zero force forever. The four tires are
    # what touches the ground, and the foot terms fold them back into two feet.
    #
    # Bodies resolve in prim-label order, which is neither this order nor upstream's, so every term
    # selects them by name with ``preserve_order=True``.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(tire|tire_2|tire_3|tire_4))?",
        history_length=3,
        track_air_time=True,
    )

    # Self-collision sensor, carrying the same documented narrowing of upstream's
    # trunk-subtree-against-itself sensor that the stand-up and roll tasks do: it senses the trunk
    # against the collider-carrying bodies below it, which reports the same 0-or-1 signal through
    # :func:`~isaaclab_tasks.contrib.microduck.pollen_mdp.rewards.self_collision_cost` but does not see, for
    # instance, one blade clipping the other. Isaac Lab resolves a per-partner force matrix only for
    # a ``prim_path`` matching a single prim per environment; widening it needs the Newton backend's
    # shape-level ``sensor_shape_prim_expr`` / ``filter_shape_prim_expr`` and is tracked separately.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/(jaw_soft|hip_l|hip_l_2|leg|leg_2|tire|tire_2|tire_3|tire_4)"
        ],
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
    """Command specifications for the MDP (addendum section 5.4).

    One command, and its three slots do not mean what the walking task's mean:

    * ``cmd[0]`` is a **throttle** -- 0 coasts, positive pushes, negative brakes -- so its range is
      asymmetric and nothing tracks it as a speed;
    * ``cmd[1]`` is pinned to zero: a skate cannot translate sideways;
    * ``cmd[2]`` is a **heading error** in radians, and its range is the clamp applied to it. At
      ``(0.0, 0.0)`` the clamp is to zero, so the heading machinery
      :class:`~isaaclab_tasks.contrib.microduck.pollen_mdp.commands.RelativeHeadingVelocityCommand` runs is
      computed and then discarded (addendum section 7.21). Upstream disabled turning while it worked
      on the stride; the term is carried across because re-enabling it is a range change.
    """

    base_velocity = mdp.RelativeHeadingVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        # Inherited from upstream's base template and never overridden there (addendum section 7.22).
        # Unlike on the stand-up and roll tasks, where the forced surge is read by nothing, here it
        # means a fifth of the resamples get a commanded throttle of at least 0.3 -- which six reward
        # terms do read. It skews the command distribution and is reproduced rather than cleaned up.
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.RelativeHeadingVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to every other MicroDuck task's: the same 14 servos in the same deploy order, at the
    same unit scale, closing the same encoder-bias loop. The four wheel hinges are **not** actions --
    the servo group's ``^(?!passive_).*`` selector leaves them out, which is the property that keeps
    the action space 14 wide on a model with 18 joints.

    There is deliberately no ``clip``: upstream tried an environment-side action clip here and
    rejected it, because the deployed runtime does not clip and the policy would then meet a bound in
    simulation that does not exist on the robot. :attr:`RewardsCfg.action_over_limit` is the
    policy-side deterrent that replaces it, and it exports with the network.
    """

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (addendum section 6).

    The actor group is the family's 61-wide deploy contract, and on this model the passive-joint
    exclusion in the two joint blocks is **load-bearing** rather than defensive: without it they
    would be 18 wide and the actor would present 69 values to a runtime expecting 61 (addendum
    section 5.1). The exclusion is spelled out as 14 names here, which cannot pick up a wheel.

    This task has neither a head-pose nor a body-pose command, so both slots are zero padding -- the
    deployed runtime sends zeros for them too.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        See :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.ObservationsCfg` for
        what each corruption models; the terms, their order, their noise and their delays are
        upstream's velocity values, which this task copies term for term.
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
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (addendum section 6.2).

        The actor's terms with every corruption removed, plus the base linear velocity, the three
        foot terms and -- unique to this task -- the four **wheel speeds**, which no sensor on the
        robot reports and which are the most direct measure of whether a stroke worked.

        The two sensor-derived terms are the NaN-guarded variants. Upstream applies them on the
        stand-up task only and gives its reason there, and the extraction reads their absence
        elsewhere as drift rather than design -- naming this task in particular, because ``wheel_vel``
        is fed by free-spinning unlimited joints, which the NaN termination's own docstring names as
        an explosion source (addendum section 7.9). See :class:`TerminationsCfg` for the matching half
        of that deviation.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_air_time = ObsTerm(
            func=mdp.foot_air_time_safe,
            params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
        )
        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
        )
        foot_contact_forces = ObsTerm(
            func=mdp.foot_contact_forces_safe,
            params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
        )
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _WHEEL_JOINT_CFG})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events (addendum section 5.6).

    The velocity task's randomization suite with three roller-specific changes:

    * ``foot_friction`` is **deleted**. Upstream's one-line reason is that *wheels roll; ground
      friction lives in the XML* -- a rolling contact does not slip, so randomizing its friction
      randomizes nothing the policy can feel.
    * :attr:`randomize_wheel_friction` is added in its place: the bearings' MJCF friction is zero for
      trainability, and the :attr:`CurriculumCfg.wheel_friction` schedule ramps a realistic drag in
      once skating is robust.
    * :attr:`randomize_armature` **excludes the wheels**. Their armature is 1e-4 and scaling it is not
      the intended bearing model -- the friction event above is.

    The push is gentler than the walking task's, ``+/-0.2 m/s`` against ``+/-0.3``: a robot on wheels
    keeps the velocity it is given.

    **Declaration order is behaviour**, since Isaac Lab fires reset events in the order they are
    declared. This is upstream's reset chain.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

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
    # Reset: redrawn every episode, in this order.
    ##

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Upstream samples the absolute base height in (0.1335, 0.1435) m; Isaac Lab samples an
            # offset from the configured default, which is that band's midpoint
            # (:data:`MICRODUCK_ROLLERS_SPAWN_HEIGHT`).
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose, wheels included.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # Wheel-bearing drag, shipped at zero and ramped by :attr:`CurriculumCfg.wheel_friction`.
    randomize_wheel_friction = EventTerm(
        func=mdp.randomize_joint_dry_friction,
        mode="reset",
        params={"asset_cfg": _WHEEL_JOINT_CFG, "friction_range": (0.0, 0.0)},
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    # Servos only. The wheels are excluded by naming the 14 driven joints, which is upstream's
    # ``^(?!passive_).*`` written out.
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    ##
    # Interval.
    ##

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (addendum section 5.3).

    Upstream keeps five terms of the mjlab base template -- ``pose``, ``upright``, ``body_ang_vel``,
    ``angular_momentum`` and ``action_rate_l2`` -- and deletes everything else, including both
    velocity-tracking terms and ``dof_pos_limits``, which every other MicroDuck task silently
    inherits. The rest of this dict is the skating recipe.

    Read it in three groups:

    * **The task.** :attr:`wheel_speed` is the only positive task reward, and :attr:`braking` is its
      mirror for a negative throttle. Nothing else pays for going anywhere.
    * **The stroke.** :attr:`skating_air_time` pays each swing and therefore drives cadence;
      :attr:`glide` pays a quiet single-support coast and therefore drives commitment;
      :attr:`single_support` rewards one blade down and charges two, which is the direct
      anti-swizzle signal; :attr:`gait_symmetry` stops the stride going lopsided;
      :attr:`feet_flat` asks the loaded blade to lie flat; :attr:`forward_lean` asks the trunk to
      lean into the push. Upstream's weights here are the record of a long tuning history and are
      copied rather than re-derived.
    * **The regularizers.** :attr:`action_over_limit` replaces the environment-side action clip
      upstream rejected, and :attr:`hip_roll_neutral` closes a stance that otherwise rests splayed on
      the hip-roll stops.
    """

    ##
    # Base-template terms upstream keeps (reference section 5).
    ##

    # ``std`` is the base template's ``sqrt(0.2)``, not the walking task's ``sqrt(0.05)``: upstream
    # narrows it for MicroDuck in the velocity recipe only, and this task re-derives from the
    # template and never revisits it.
    upright = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.2), "asset_cfg": _TRUNK_BODY_CFG},
    )

    # The hip-roll tolerances are the loosened ones -- 0.6 walking and 0.8 running against the
    # walking task's 0.05 -- because a skating push *is* a wide lateral excursion of the hips. The
    # running regime is genuinely reachable here: the threshold is 0.5 rather than the template's 1.5.
    #
    # The selection is all 18 joints and the wheels' tolerance is 999, so their contribution to the
    # mean is 1 whatever angle they have spun to. Narrowing the selection instead would change the
    # value of the mean (addendum section 7.20).
    pose = RewTerm(
        func=mdp.pose_mode_switch,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {
                ".*hip_yaw.*": 0.05,
                ".*hip_roll.*": 0.05,
                ".*hip_pitch.*": 0.05,
                ".*knee.*": 0.05,
                ".*ankle.*": 0.05,
                ".*neck.*": 0.05,
                ".*head.*": 0.05,
                ".*passive_.*": 999.0,
            },
            "std_walking": {
                ".*hip_yaw.*": 0.3,
                ".*hip_roll.*": 0.6,
                ".*hip_pitch.*": 0.4,
                ".*knee.*": 0.4,
                ".*ankle.*": 0.25,
                ".*neck.*": 0.05,
                ".*head.*": 0.05,
                ".*passive_.*": 999.0,
            },
            "std_running": {
                ".*hip_yaw.*": 0.5,
                ".*hip_roll.*": 0.8,
                ".*hip_pitch.*": 0.8,
                ".*knee.*": 0.8,
                ".*ankle.*": 0.5,
                ".*neck.*": 0.05,
                ".*head.*": 0.05,
                ".*passive_.*": 999.0,
            },
            "walking_threshold": 0.01,
            "running_threshold": 0.5,
            "asset_cfg": _ALL_JOINT_CFG,
        },
    )

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # Ramped to -2.0 by iteration 500, which is upstream's main lever for a calmer stride.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    ##
    # Posture and height.
    ##

    # KNOWN-STALE BAND, REPRODUCED DELIBERATELY. This robot stands 0.1407 m tall
    # (:data:`MICRODUCK_ROLLERS_STANDING_HEIGHT`), so at weight 2.0 this term asks it to crouch 1.7 to
    # 4.7 cm below its own equilibrium from the very first step. The two numbers are sized against the
    # wheel-less stand-up model this environment used to load by mistake, whose stand height of
    # 0.11718 m sits neatly inside them: git-dating the extraction puts the band at 2026-03-11 and the
    # roller-model wiring at 2026-07-09 (addendum section 7.16).
    #
    # It is reproduced verbatim anyway, on the same parity-first grounds as the head-pose caps in the
    # walking port: upstream's deployed skating policies were trained with this band, permanent crouch
    # and all, so a re-measured band would be a different task. Re-measuring is a deliberate retune
    # with its own training run, not a port fix.
    com_height_target = RewTerm(
        func=mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )

    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG},
    )

    # Gated per foot by that foot's own contact, so the swing blade is free to tilt. Upstream's
    # earlier ungated version was minimized by keeping both blades flat on the floor -- the swizzle --
    # and actively fought the stride.
    feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": _FOOT_BODY_CFG,
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "normal_axis": MICRODUCK_FOOT_NORMAL_AXIS,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )

    ##
    # Head and effort regularizers.
    ##

    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    neck_joint_pos_l2 = RewTerm(func=mdp.joint_pose_l2, weight=-0.5, params={"asset_cfg": _HEAD_JOINT_CFG})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})
    action_over_limit = RewTerm(
        func=mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    hip_roll_neutral = RewTerm(func=mdp.joint_deviation_l1, weight=-2.0, params={"asset_cfg": _HIP_ROLL_JOINT_CFG})

    ##
    # The task: spin the wheels, and stop when asked.
    ##

    wheel_speed = RewTerm(
        func=mdp.wheel_speed_reward,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": _WHEEL_JOINT_CFG,
            # the ground speed the tanh saturates near. Upstream lowered it from 0.5 after measuring
            # a trained policy topping out at about 0.33 m/s: a target above the achievable speed sits
            # on the un-saturated slope and keeps asking for more, which is how the launch
            # instability got trained in.
            "vel_scale": 0.3,
            # KNOWN-WRONG RADIUS, REPRODUCED DELIBERATELY. The measured tire radius on this model is
            # 0.0150 m; upstream never overrides the reward's 0.0175 m default, so its ``tanh`` is
            # scaled to 17.14 rad/s, which at the true radius is a ground speed of 0.257 m/s rather
            # than the intended 0.300 (addendum section 7.15). Same parity-first grounds as the
            # ``com_height_target`` band above: the deployed policies were trained against this
            # saturation point, and correcting it is a retune rather than a port fix.
            "wheel_radius": 0.0175,
        },
    )
    braking = RewTerm(
        func=mdp.braking_reward,
        weight=1.0,
        params={"command_name": "base_velocity", "vel_std": 0.3},
    )

    ##
    # The stroke.
    ##

    skating_air_time = RewTerm(
        func=mdp.skating_air_time_reward,
        weight=1.5,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )
    # Weighted above the air-time term on purpose: the two pull against each other -- cadence against
    # commitment -- and upstream tilted the balance toward the glide after the stride stayed frantic.
    glide = RewTerm(
        func=mdp.glide_reward,
        weight=4.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _LEG_JOINT_CFG,
            "vel_ref": 0.2,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )
    single_support = RewTerm(
        func=mdp.single_support_reward,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "vel_gate_ref": 0.2,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )
    gait_symmetry = RewTerm(
        func=mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
    )
    # ``target_pitch`` is the sine of the lean, so 0.262 asks for 15.2 degrees nose-down. Upstream's
    # docstring for this kernel describes the negated quantity and its code does not negate; the code
    # is what is ported, and it is the sign that makes a positive target a *forward* lean (addendum
    # section 7.17).
    forward_lean = RewTerm(
        func=mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "base_velocity", "target_pitch": 0.262, "std": 0.1},
    )
    # The only thing keeping a straight-line skater straight, since the heading command is clamped to
    # zero. Corrective rather than a yaw-rate penalty, which upstream tried and reverted: freezing the
    # yaw rate made the drift worse, because the policy could then no longer steer back.
    heading_hold = RewTerm(func=mdp.heading_hold_reward, weight=1.0, params={"std": 0.4})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (addendum section 5.5).

    Unlike the stand-up and roll tasks, this one keeps the walking recipe's tilt termination: falling
    over is a failure here, not the task. Upstream's inherited terrain-bounds check -- all-false on a
    ground plane, on a flat-only task -- is not carried over (addendum section 7.24).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # Catches a *broken* robot rather than a fallen one.
    #
    # Deviation from upstream, deliberately: upstream leaves this term's sensor list empty here and
    # names its foot contact sensor only on the stand-up task, which the extraction reads as drift and
    # recommends closing everywhere -- naming this task specifically, because the critic's
    # ``wheel_vel`` observation is fed by free-spinning unlimited joints (addendum section 7.9). The
    # guard only changes behaviour in states that are already broken, and the matching half of it is
    # the pair of NaN-safe critic terms in :class:`ObservationsCfg`.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (addendum section 5.7).

    Upstream deletes the base template's ``terrain_levels`` and ``command_vel`` schedules, and this
    task has no head- or body-pose command to open, so what is left is one reward ramp, one physical
    ramp and the two centre-of-mass ramps the whole family shares.

    Upstream's schedules are written in PPO iterations; :func:`_iterations` converts them to the
    global environment-step count these terms compare against.
    """

    # Upstream's main "less movement" lever, raised from the walking task's ceiling of -1.0 to -2.0:
    # it prices fast and large action changes, so rapid alternation -- the frantic kick cadence -- is
    # what it hits hardest.
    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -1.0},
                {"step": _iterations(250), "weight": -1.5},
                {"step": _iterations(500), "weight": -2.0},
            ],
        },
    )

    # Bearing drag, held at zero until skating is robust. Upstream's earlier schedule started adding
    # drag at iteration 750 -- exactly when the wheel-speed reward peaked -- and pushed the policy off
    # skating into a heading-farming local optimum, so both the onset and the ceiling were relaxed.
    #
    # ``event_param_stages`` with the exclusive comparison, which is what upstream's dedicated
    # ``wheel_friction_curriculum`` uses (addendum section 7.6).
    wheel_friction = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "randomize_wheel_friction",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"friction_range": (0.0, 0.0)}},
                {"step": _iterations(2000), "params": {"friction_range": (5e-4, 5e-4)}},
                {"step": _iterations(3500), "params": {"friction_range": (1e-3, 1e-3)}},
                {"step": _iterations(5000), "params": {"friction_range": (1.5e-3, 1.5e-3)}},
            ],
        },
    )

    # Capped at +/-10 mm rather than the walking task's +/-15 mm: skating is less forgiving of a
    # displaced centre of mass than walking is.
    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
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
class MicroDuckVelocityRollersFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck roller-skating velocity environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckRollersPhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
    scene: MicroDuckRollersSceneCfg = MicroDuckRollersSceneCfg(num_envs=4096, env_spacing=2.0)
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
        # at, and 20 s episodes are 1000 control steps -- the velocity task's, which upstream leaves
        # untouched here (reference section 1).
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, as every other MicroDuck task does and
        # as upstream does. The decimation above is even, which is what lets the stateful servo delay
        # line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        for sensor in (self.scene.contact_forces, self.scene.self_collision):
            if sensor is not None:
                sensor.update_period = self.sim.dt
        # MicroDuck stands 0.14 m tall on skates, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
