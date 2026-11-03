# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controlled descent on a procedural roller-skating slope.

Difficulty changes the incline angle. The robot starts on the ramp and uses relative-heading
commands, wheel motion and balance rewards to control its descent.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.rollerslope.terrain import MICRODUCK_RAMP_DEG_MAX, FlatRampTerrainCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_FOOT_NORMAL_AXIS,
    MICRODUCK_TIRE_BODY_NAMES,
    MICRODUCK_TIRES_PER_FOOT,
    MICRODUCK_WHEEL_JOINT_NAMES,
    MicroDuckRollersSceneCfg,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import CommandsCfg as RollersCommandsCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import EventsCfg as RollersEventsCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import TerminationsCfg as RollersTerminationsCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_TRUNK_BODY_NAME,
)

##
# The slope
##

MICRODUCK_SLOPE_TILE_SIZE = (15.0, 4.0)
"""Size [m] of one terrain tile.

Sized by the sub-terrain it holds: the longest ramp plus its two platforms is
``2.0 + 8.0 + 4.0 = 14.0`` m, so 15 m leaves a metre of slack along the slope. On the *shortest*
3 m ramp the solid geometry ends 6 m short of the next row, which is the void
:attr:`TerminationsCfg.fell_into_void` exists for.
"""

MICRODUCK_SLOPE_NUM_ROWS = 10
"""Rungs on the difficulty ladder. Row ``r`` draws its angle in ``[2 + 1.8r, 3.8 + 1.8r]`` degrees."""

_FLAT_RAMP_CFG = FlatRampTerrainCfg()
"""The sub-terrain at defaults; the generator overwrites its ``size`` with the tile's."""

MICRODUCK_SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=MICRODUCK_SLOPE_TILE_SIZE,
    curriculum=True,
    num_rows=MICRODUCK_SLOPE_NUM_ROWS,
    num_cols=1,
    difficulty_range=(0.0, 1.0),
    sub_terrains={"flat_ramp": _FLAT_RAMP_CFG},
)
"""The slope ladder MicroDuck descends.

The tile is collided against as a **heightfield** rather than as the raw triangle mesh the generator
emits; see :attr:`~isaaclab_tasks.contrib.microduck.rollerslope.terrain.FlatRampTerrainCfg.convert_to_heightfield`
for why. The rasterization resolution is this generator's ``horizontal_scale``, left at the stock
0.1 m: halving it to 0.05 m moves the trunk height on a 20 degree ramp by under a millimetre, so the
default costs nothing.

Note:
    The per-tile ramp length is drawn from the **global** numpy random state, because Isaac Lab
    sub-terrain functions take no generator and the terrain generator seeds only its own. Pin
    :attr:`~isaaclab_tasks.contrib.microduck.rollerslope.terrain.FlatRampTerrainCfg.ramp_length_range`
    to a single value to make a run reproducible; the accuracy-gate regimes do exactly that.
"""

MICRODUCK_SLOPE_VOID_FLOOR = -(
    _FLAT_RAMP_CFG.ramp_length_range[1] * math.tan(math.radians(MICRODUCK_RAMP_DEG_MAX)) + 0.5
)
"""World-frame height [m] below which the robot has left the terrain: **-3.41176 m**."""

MICRODUCK_SLOPE_FELL_OVER_LIMIT = 1.0
"""Trunk tilt [rad] that ends the episode: 57.3 degrees, tightened from the skating task's 70."""

MICRODUCK_SLOPE_ENTRY_SPEED_X = (0.25, 0.45)
"""Forward speed [m/s] the robot leaves the reset already rolling at."""

MICRODUCK_SLOPE_UPRIGHT_STD = math.sqrt(0.08)
"""Width [-] of the ungated upright reward Gaussian."""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_WHEEL_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_TIRE_BODY_NAMES, preserve_order=True)

##
# Scene definition
##


@configclass
class MicroDuckRollerSlopeSceneCfg(MicroDuckRollersSceneCfg):
    """The skating scene on generated slope terrain.

    One field changes. The robot, the tire contact sensor, the self-collision sensor and the lighting
    are the skating scene's, and the physics material is the same neutral one -- the friction the
    tires roll on is authored on them in the model, not on the ground.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MICRODUCK_SLOPE_TERRAIN_CFG,
        # every environment starts on the gentlest row and is promoted from there by
        # :attr:`CurriculumCfg.terrain_levels`
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class CommandsCfg(RollersCommandsCfg):
    """The skating command term, kept alive and neutralized.

    The term is **not** deleted, and that is deliberate on both sides: the 61-wide observation
    reserves three slots for the twist, and removing the term would collapse the deployed vector to
    58. So the throttle range is zeroed and every environment is made a standing one, which is what
    the base term zeroes on every step.

    The descent comes from gravity, so there is nothing to command; the skating task's throttle would
    only ask the policy to pedal on a hill it is already falling down.
    """

    def __post_init__(self):
        command = self.base_velocity
        command.rel_standing_envs = 1.0
        command.rel_heading_envs = 0.0

        command.rel_forward_envs = 0.0
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    What is left reads in three groups:

    * **The task.** :attr:`wheel_glide` is the only positive task reward and it is capped, one-sided
      and measured on the wheels, so nothing pays for dropping faster, rolling backwards is free
      rather than charged, and running down on the blades earns nothing. :attr:`heading_hold` keeps
      the descent pointed down the fall line, and :attr:`alive` pays for not ending the episode.
    * **The posture.** :attr:`upright` at weight 3.0 is the heaviest term in the dict -- staying on
      the wheels *is* the task -- and :attr:`feet_flat` asks the loaded blade to lie flat on the
      incline.
    * **The regularizers**, which are the skating task's, at the skating task's weights except one.

    No curriculum touches any of these weights (see :class:`CurriculumCfg`), so unlike on the skating
    task the declared literals are the live values from step 0.
    """

    ##
    # The task: keep rolling, keep pointing down the hill, keep the episode alive.
    ##

    wheel_glide = RewTerm(
        func=mdp.wheel_glide_reward,
        weight=2.0,
        params={"asset_cfg": _WHEEL_JOINT_CFG, "cap_speed": 0.35},
    )
    # The same kernel and width the skating task uses, at 1.5 rather than 1.0: there is no throttle
    # to steer with here, so drifting off the fall line is not recoverable by pushing harder.
    heading_hold = RewTerm(func=mdp.heading_hold_reward, weight=1.5, params={"std": 0.4})
    # One step per episode at weight 1.0 -- the same substitution the rest of the family makes.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    ##
    # Posture on the incline.
    ##

    # The heaviest term in the dict.
    upright = RewTerm(
        func=mdp.upright_selected_body,
        weight=3.0,
        params={"std": MICRODUCK_SLOPE_UPRIGHT_STD, "asset_cfg": _TRUNK_BODY_CFG},
    )

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
    # Regularizers.
    ##

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    # Hold the head near its nominal pose during a descent the robot cannot steer out of.
    neck_joint_pos_l2 = RewTerm(func=mdp.joint_pose_l2, weight=-0.75, params={"asset_cfg": _HEAD_JOINT_CFG})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class EventsCfg(RollersEventsCfg):
    """The skating randomization suite plus the rolling entry.

    The domain randomization is the skating task's, term for term and range for range. Two things
    change, and both are about how an episode starts.
    """

    # Appended last, which is what puts it after ``reset_base``. It names the four wheels rather than
    # taking the default selection: the event spins whatever it is given at ``v / r``, and a default
    # ``SceneEntityCfg("robot")`` selects every hinge as a slice, which would launch the servos at
    # tens of radians a second.
    reset_rolling_entry = EventTerm(
        func=mdp.reset_rolling_entry,
        mode="reset",
        params={"asset_cfg": _WHEEL_JOINT_CFG, "speed_range": MICRODUCK_SLOPE_ENTRY_SPEED_X},
    )

    def __post_init__(self):
        # Face down the slope. Everything else about the base reset is the skating task's, including the +/-0.5 m
        # horizontal jitter -- which on an origin that sits on the incline can spawn the trunk up to 109 mm
        # inside the starting platform or 182 mm above the ramp at 20 degrees, against a standing height of
        # 138 mm.
        self.reset_base.params["pose_range"] = {**self.reset_base.params["pose_range"], "yaw": (0.0, 0.0)}

        self.reset_base.params["velocity_range"] = {}


@configclass
class TerminationsCfg(RollersTerminationsCfg):
    """The skating terminations, tightened, plus the void guard.

    ``time_out`` and ``nan_state`` are inherited untouched. ``fell_over`` is redeclared at a tighter
    limit angle, and one termination is added.
    """

    # The only height quantity in the whole task, and an **absolute world-z** one: it is deliberately independent of
    # the ramp a given environment drew, so it cannot fire during a nominal descent and only fires once the
    # robot has left the solid past a short ramp.
    fell_into_void = DoneTerm(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"minimum_height": MICRODUCK_SLOPE_VOID_FLOOR},
    )

    def __post_init__(self):
        self.fell_over.params["limit_angle"] = MICRODUCK_SLOPE_FELL_OVER_LIMIT


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Not the stock ``terrain_levels_vel``: that one demotes against the distance a *commanded*
    # velocity should have covered, and this task's command is zero, so every environment would be
    # demoted on every reset. Progress is the raw signed distance down the slope instead -- promote
    # past 6.0 m, demote below 3.0 m on the 15 m tile.
    terrain_levels = CurrTerm(func=mdp.terrain_levels_slope)


##
# Environment configuration
##


@configclass
class MicroDuckRollerSlopeFlatEnvCfg(MicroDuckVelocityRollersFlatEnvCfg):
    """MicroDuck slope-descent environment on generated ramp terrain."""

    scene: MicroDuckRollerSlopeSceneCfg = MicroDuckRollerSlopeSceneCfg(num_envs=4096, env_spacing=2.0)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # the terrain-level curriculum needs the generator to lay its rows out by difficulty, so
        # removing the ladder -- as an ablation would -- also removes the ordering it relies on
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None

        # Reserve contact and constraint headroom for simultaneous body-ground contact.

        # Constraint limits are per environment; contact storage is shared across environments.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 448
        newton_mjwarp.solver_cfg.nconmax = 112
        self.sim.physics.default = newton_mjwarp
