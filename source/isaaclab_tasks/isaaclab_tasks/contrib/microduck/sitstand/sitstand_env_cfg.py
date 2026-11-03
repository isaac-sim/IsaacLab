# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commanded transitions between seated and standing postures.

A slewed posture command drives pose, height and stillness targets. Seated perturbations
and neck-strain costs train robust transitions with whole-body contact.

Includes Apache-2.0 adaptations from Pollen Robotics; see ``UPSTREAM_NOTICE.md``.
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

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_CFG
from isaaclab_tasks.contrib.microduck.standup.standup_env_cfg import (
    MICRODUCK_SIT_HEIGHT,
    MICRODUCK_SITTING_JOINT_POS,
    MICRODUCK_STAND_HEIGHT,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# The two postures
##

MICRODUCK_POSTURE_DWELL_S = (3.5, 6.5)
"""Time [s] the robot is left in a commanded posture before the request may flip.

The lower bound has to clear a gentle transition -- about 1.5 s -- plus a moment of rest, so
"arrive, then hold still" is trained on every segment rather than only on the lucky long ones.
"""

MICRODUCK_SIT_PROBABILITY = 0.5
"""Probability that a posture resample requests the sit.

Drawn independently of the reset posture, so the four (start state x request) combinations -- sit
from standing, rise from seated, hold the stand, hold the sit -- get equal coverage.
"""

MICRODUCK_POSTURE_RAMP_S = 2.0
"""Time [s] the reward's target blend takes to traverse the full stand-to-sit range.

The anti-crash mechanism; see the module docstring. It is also the sanity check on the speed caps
below: 55 mm over 2 s is about 0.028 m/s, comfortably under both of them.
"""

MICRODUCK_STAND_UPRIGHT_HEIGHT = 0.10
MICRODUCK_SIT_UPRIGHT_HEIGHT = 0.075
"""Trunk heights [m] the ``upright_while_tall`` gate opens and closes over.

Full upright incentive above the first, fading to nothing below the second, where a trunk resting on
its base is a legitimate seated pose. The window is what denies the "tip backward while still high"
descent, which would otherwise farm the height rewards through a controlled fall; the always-on
linear upright floor covers the seated regime.
"""

MICRODUCK_MAX_DESCENT_SPEED = 0.05
MICRODUCK_MAX_RISE_SPEED = 0.08
"""Vertical trunk speeds [m/s] the two cap penalties start charging above.

The rise cap is the looser of the two because rising against gravity needs a brief burst to rock the
weight over the heels, and it is introduced by curriculum only once the rise exists at all.
"""

MICRODUCK_RISE_BOOTSTRAP_CEILING = 0.125
"""Trunk height [m] above which the rise bootstrap stops paying."""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, velocity-matched value."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency."""


def _iterations(count: int) -> int:
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckSitStandPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck sit-stand environment."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Reserve contact and constraint headroom for simultaneous body-ground contact.
            # Reserve contact and constraint headroom for simultaneous body-ground contact.
            njmax=128,
            nconmax=32,
            # See the class docstring: this is the seated contact-NaN fix.
            iterations=30,
            ls_iterations=50,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckSitStandSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the robot pushes off is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The seated pose rests the trunk on the floor and may put the knees and the head there too on the way, so the
    # walking model's two soles are not enough: its trunk would sink through the plane.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
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
    """Command specifications for the MDP.

    Two live commands and one shape placeholder. The twist slot carries the posture flag instead of
    a velocity, which is the task; the head keeps the family's pose-delta command, so the robot can
    still be asked to look around in either posture; and the six-wide body-pose slot is zero padding,
    filled in :class:`ObservationsCfg` rather than by a command term.
    """

    # The posture flag, riding in the twist slot so the deployed 61-wide vector keeps its shape. The
    # inherited velocity ranges are never sampled -- the term writes the flag directly -- and are
    # therefore not configured; the resampling time is the dwell in each posture.
    base_velocity = mdp.SitStandCommandCfg(
        asset_name="robot",
        resampling_time_range=MICRODUCK_POSTURE_DWELL_S,
        heading_command=False,
        debug_vis=False,
        sit_prob=MICRODUCK_SIT_PROBABILITY,
        ramp_s=MICRODUCK_POSTURE_RAMP_S,
        sit_height=MICRODUCK_SIT_HEIGHT,
        stand_height=MICRODUCK_STAND_HEIGHT,
        ranges=mdp.SitStandCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )

    # Joint-position deltas from the stand pose for (neck_pitch, head_pitch, head_yaw, head_roll).
    # Identical to the velocity and stand-up tasks', including the ``head_pose_range`` curriculum
    # that opens it: the head is commandable in *both* postures here, which is why the composite goal
    # score carries a head factor rather than leaving the head to the light tracking term alone.
    head_pose = mdp.ZeroablePoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to the velocity task's: the same 14 servos in the same deploy order, at the same unit
    scale, closing the same encoder-bias loop.
    """

    joint_pos = mdp.ArticulationEncoderBiasActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    The actor group is byte-for-byte the velocity task's 61-wide deploy contract, which is the whole
    point of the family: one runtime on the robot feeds every MicroDuck policy from the same buffer.

    **The policy observes the raw posture flag, not the slewed blend the rewards track.** That is
    the deployment contract -- a button press flips the flag instantly -- and it is what makes the
    trained response to a flip a smooth two-second glide rather than a step.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract."""

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
            func=mdp.joint_pos_rel_articulation_biased,
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
        # Shape placeholder for the deployed vector's body-pose slot, which this task has no command
        # for. It is deliberately constant zero rather than a live tiny range: the slot exists so a
        # runtime can hot-swap this policy for one that does steer the trunk.
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_articulation_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False}
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_air_time = ObsTerm(func=mdp.foot_group_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_group_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_group_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events.

    The domain-randomization suite is the velocity task's, term for term and range for range, so a
    policy trained here meets the same hardware spread as one trained to walk. What this task adds is
    a two-bucket :attr:`set_ground_state`: half the episodes start standing and half start already
    seated, drawn independently of the first posture request.

    **Declaration order is behaviour.** Isaac Lab fires reset events in the order they are declared,
    and :attr:`set_ground_state` overwrites the root height and orientation that :attr:`reset_base`
    wrote. Only the root reset's horizontal spread survives. Reordering these reset events changes
    the spawn distribution.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

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

    encoder_bias = EventTerm(
        func=mdp.randomize_articulation_encoder_bias,
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

    # Only the horizontal spread is live, so only the horizontal spread is configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # The episode distribution, and the reason this task trains four skills rather than two: the
    # starting posture is drawn here, the first request is drawn by the command term, and the two are
    # independent. No curriculum touches it -- the mix is 50/50 from the first step to the last.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.5,
            "standing_prob": 0.5,
            # the seated keyframe settles at 0.060 m, so this band spawns it at or just above its
            # own rest rather than dropping it
            "sitting_z_range": (0.06, 0.075),
            "standing_z_range": (0.11, 0.12),
            "sitting_joint_pos": MICRODUCK_SITTING_JOINT_POS,
            # about 6 degrees per joint. Without it the policy overfits the exact keyframe and does
            # not survive a hand-off from the real sit it is supposed to continue from.
            "sitting_joint_noise_std": 0.10,
            "sitting_tilt_max": math.radians(8.0),
            "asset_cfg": _SERVO_JOINT_CFG,
        },
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

    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    ##
    # Interval.
    ##

    # Ramped from nothing by the ``push_magnitude`` curriculum, and ramped *later* here than anywhere else in the
    # family.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )

    # Starting the disturbances at zero protects transition discovery; the live curricula below introduce them.
    seated_head_jerk = EventTerm(
        func=mdp.seated_head_command_jerk,
        mode="interval",
        interval_range_s=(0.3, 0.8),
        params={"prob": 0.0, "flip_frac": 0.5},
    )
    seated_tip = EventTerm(
        func=mdp.seated_backward_tip,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"ang_vel_range": (0.0, 0.0), "prob": 0.5, "lateral_frac": 0.3},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Every task term reads the commanded posture and selects its target from it, so one stack rewards
    both rest states and both transitions. The shapes recur and explain the apparent duplication:

    * **Two-layer attractors.** ``posture_height`` and ``posture_height_sharp`` are the same quantity
      at a wide and a narrow width; the wide layer saturates well before the robot has arrived, and
      the narrow one supplies the last few millimetres of gradient.
    * **L1 drivers.** ``posture_pose_l1`` and ``posture_height_l1`` carry a constant gradient where
      the Gaussians are flat, which is what makes resting in the *wrong* posture net-negative in both
      directions rather than merely unrewarded.
    * **Multiplicative goal.** ``posture_composite`` scores height, tilt, pose and head as a product,
      so the partial-sum compromises the rest of the stack invites -- plank, flop, lean, park a
      centimetre short -- collapse to nothing.
    """

    ##
    # Posture: hold the legs and the trunk at the commanded target.
    ##

    # Generous width on purpose: the knee travels about 1.35 rad between the two keyframes, so a
    # tight kernel would be flat across most of a transition.
    posture_pose_legs = RewTerm(
        func=mdp.posture_pose_gaussian,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "sit_joint_pos": MICRODUCK_SITTING_JOINT_POS,
            "std": 0.5,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )
    posture_pose_l1 = RewTerm(
        func=mdp.posture_pose_l1,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sit_joint_pos": MICRODUCK_SITTING_JOINT_POS,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    ##
    # Head: steered by command in both postures, not folded into the posture keyframe.
    ##

    # Light on purpose, so a transient head-assist during a transition costs only a little tracking
    # reward. What stops the head being *left* down is the composite's head factor below.
    head_pose_tracking = RewTerm(
        func=mdp.head_pose_tracking_raw,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
    )

    ##
    # Height: track the commanded posture's trunk height.
    ##

    posture_height = RewTerm(
        func=mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sit_height": MICRODUCK_SIT_HEIGHT,
            "stand_height": MICRODUCK_STAND_HEIGHT,
            "std": 0.04,
        },
    )
    posture_height_sharp = RewTerm(
        func=mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sit_height": MICRODUCK_SIT_HEIGHT,
            "stand_height": MICRODUCK_STAND_HEIGHT,
            "std": 0.015,
        },
    )

    posture_height_l1 = RewTerm(
        func=mdp.posture_height_l1,
        weight=6.0,
        params={
            "command_name": "base_velocity",
            "sit_height": MICRODUCK_SIT_HEIGHT,
            "stand_height": MICRODUCK_STAND_HEIGHT,
        },
    )

    ##
    # Gentleness, in both directions.
    ##

    # Pays for the upward motion itself while a stand is requested, which destination rewards cannot:
    # they have zero gradient at zero motion, and without this the stand-up task parked seated. It
    # reads the raw flag rather than the slewed blend, so it switches off the instant a sit is asked
    # for and can never bid against the descent.
    rise_bootstrap = RewTerm(
        func=mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "max_height": MICRODUCK_RISE_BOOTSTRAP_CEILING,
            "max_vz": MICRODUCK_MAX_RISE_SPEED,
        },
    )
    # Live at full strength from step 0 and tightened to 20.0 at iteration 500.
    descent_speed = RewTerm(
        func=mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={"max_down_vel": MICRODUCK_MAX_DESCENT_SPEED, "asset_cfg": _TRUNK_BODY_CFG},
    )
    # The mirror cap, at weight zero until iteration 1500. The timing is the point: a motion tax live
    # while the rise is still being discovered makes every attempt net-negative and the skill is never
    # found.
    rise_speed = RewTerm(
        func=mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={"max_up_vel": MICRODUCK_MAX_RISE_SPEED, "asset_cfg": _TRUNK_BODY_CFG},
    )
    gentle_motion = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.05, params={"asset_cfg": _TRUNK_BODY_CFG})

    ##
    # Orientation and the composite goal state.
    ##

    # The always-on floor: at 2.5, lying on your back trails an upright rest by about 4.5 per step.
    upright_linear = RewTerm(func=mdp.body_upright_linear, weight=2.5, params={"asset_cfg": _TRUNK_BODY_CFG})
    # The height-gated booster, which denies the "tip backward while still tall" descent and doubles
    # as an arrival-uprightness pull during the rise.
    upright_while_tall = RewTerm(
        func=mdp.upright_linear_at_height,
        weight=1.5,
        params={
            "height_low": MICRODUCK_SIT_UPRIGHT_HEIGHT,
            "height_high": MICRODUCK_STAND_UPRIGHT_HEIGHT,
            "asset_cfg": _TRUNK_BODY_CFG,
        },
    )
    # "Arrive, then rest quietly and upright" as an explicit positive peak, triple-gated so it pays
    # for none of the three things that look like it: stopping half-way, resting tilted, or holding
    # still while the setpoint is still moving.
    posture_stillness = RewTerm(
        func=mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sit_height": MICRODUCK_SIT_HEIGHT,
            "stand_height": MICRODUCK_STAND_HEIGHT,
            "band_full": 0.012,
            "band_zero": 0.03,
            "vel_std": 0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )
    # The widths are the stand-up task's calibration: an ``upright_std`` of 0.40 is about 23 degrees
    # effective, so a plank at 70 degrees or more scores about zero, and a ``head_std`` of 0.40 puts
    # a fully dropped head at a factor of about 0.01.
    posture_composite = RewTerm(
        func=mdp.posture_composite,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sit_joint_pos": MICRODUCK_SITTING_JOINT_POS,
            "sit_height": MICRODUCK_SIT_HEIGHT,
            "stand_height": MICRODUCK_STAND_HEIGHT,
            "height_std": 0.03,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "head_std": 0.40,
            "head_command_name": "head_pose",
            "asset_cfg": _LEG_JOINT_CFG,
            "head_asset_cfg": _HEAD_JOINT_CFG,
        },
    )

    ##
    # Sim-to-real regularizers, matched to the velocity task.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Ramped to -1.0 by iteration 1500.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # Weight 0.0, phased in from iteration 750 once both transition motions exist. Penalizes torque
    # *change*, not magnitude, so it damps jitter without taxing a slow large motion.
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})

    self_collisions = RewTerm(
        func=mdp.self_collision_cost_saturable,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )

    # A nonnegative cost with negative weight, unlike the self-negating speed penalties above. The backward-lean
    # gate leaves head-assisted forward descents free; the alpha blend fades it during a rise.
    neck_strain_leaned = RewTerm(
        func=mdp.neck_torque_when_tilted,
        weight=-0.5,
        params={
            "asset_cfg": _HEAD_JOINT_CFG,
            "command_name": "base_velocity",
            "tilt_start_deg": 12.0,
            "tilt_full_deg": 25.0,
            "torque_scale": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    The tilt termination the velocity task ends a fall with is **absent**, and that is the design
    rather than an inheritance accident: a wobble or a tip during a transition has to play out so the
    policy pays the impact and uprightness costs, where truncating the episode would hide them.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Catches a *broken* robot rather than a fallen one. The guard only changes behaviour in states that are already
    # broken, and the matching half of it is the pair of NaN-safe critic terms in :class:`ObservationsCfg`.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan_optional_sensors,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Nothing schedules the task itself: both transitions are rewarded from step 0 and the reset mix
    never moves. What is scheduled is the *cost* of moving, and the ordering is the design:
    """

    ##
    # The two speed caps.
    ##

    descent_speed_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "descent_speed",
            "weight_stages": [
                {"step": _iterations(0), "weight": 10.0},
                {"step": _iterations(500), "weight": 20.0},
            ],
        },
    )

    rise_speed_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "rise_speed",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(1500), "weight": 5.0},
                {"step": _iterations(2500), "weight": 10.0},
            ],
        },
    )

    ##
    # Smoothness.
    ##

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

    torque_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(750), "weight": -5e-4},
                {"step": _iterations(1250), "weight": -1e-3},
            ],
        },
    )

    ##
    # Domain randomization ramps, matched to the velocity task.
    ##

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

    head_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": _iterations(0), "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": _iterations(500), "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))},
                {"step": _iterations(1000), "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))},
                {"step": _iterations(1500), "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))},
                {"step": _iterations(2000), "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )

    # The latest push ramp in the family, for the reason ``push_robot`` gives.
    push_magnitude = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "push_robot",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}},
                {"step": _iterations(1000), "params": {"velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}}},
                {"step": _iterations(1500), "params": {"velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}}},
                {"step": _iterations(2000), "params": {"velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}}},
                {"step": _iterations(2500), "params": {"velocity_range": {"x": (-0.30, 0.30), "y": (-0.30, 0.30)}}},
            ],
        },
    )

    # Start only after sitting and rising exist, alongside widening the head-command ranges.
    # These use event_param_curriculum's inclusive boundary, unlike push_magnitude above.
    seated_head_jerk_prob = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "seated_head_jerk",
            "param_stages": [
                {"step": _iterations(0), "params": {"prob": 0.0}},
                {"step": _iterations(1500), "params": {"prob": 0.2}},
                {"step": _iterations(2000), "params": {"prob": 0.4}},
            ],
        },
    )
    seated_tip_range = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "seated_tip",
            "param_stages": [
                {"step": _iterations(0), "params": {"ang_vel_range": (0.0, 0.0)}},
                {"step": _iterations(1500), "params": {"ang_vel_range": (1.0, 2.5)}},
                {"step": _iterations(2000), "params": {"ang_vel_range": (1.5, 3.25)}},
                {"step": _iterations(2500), "params": {"ang_vel_range": (2.0, 4.0)}},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckSitStandFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck sit-stand environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckSitStandPhysicsCfg())

    scene: MicroDuckSitStandSceneCfg = MicroDuckSitStandSceneCfg(num_envs=4096, env_spacing=2.0)
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
        # at. Episodes are 12 s -- 600 control steps -- sized so that two or three posture segments
        # at the 3.5-6.5 s dwell fit inside one, which is at least one full sit, rest, rise and rest
        # cycle per episode.
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        # The decimation above is even, which is what lets the stateful servo delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
