# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slope-descent environment for the roller-skating Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks4 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the skating recipe live in the companion
``historical tasks2 source extraction`` and are cited as "roller section N".

The robot is put on a ramp with a commanded velocity of exactly zero and asked to ride it down. It
cannot pedal -- the four wheels are undriven, as on every roller task -- so the only thing that moves
it is gravity, and the only thing it controls is whether it stays upright long enough to keep
rolling. That is the whole task, and three consequences shape the recipe:

* **The terrain is the task.** :data:`MICRODUCK_SLOPE_TERRAIN_CFG` lays ten rows of
  flat-plus-ramp-plus-runout tile, from a 2 degree slope to a 20 degree one, and returns each row's
  spawn origin **on the incline**. :attr:`CurriculumCfg.terrain_levels` is what promotes a robot that
  rode its ramp out to a steeper one.
* **Nothing measures height against the environment origin.** The skating recipe already deleted
  every sensor-backed height term the mjlab template adds, and this task then deletes every
  inherited reward but ``action_rate_l2``, so no observation, reward or termination is
  terrain-relative. The only height quantity anywhere is
  :attr:`TerminationsCfg.fell_into_void`, an **absolute world-z** guard. This is why the task needs
  correct per-environment origins rather than a terrain-height sensor, and it is what let the port
  ship without one.
* **The actor is slope-blind.** The 61-wide observation is the skating task's, unchanged: no height
  scan, no terrain sensor, no slope angle. The incline reaches the policy through
  ``projected_gravity``, ``base_ang_vel`` and proprioception only.

Like the two trick tasks, this environment is built as a delta on
:class:`~isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg.MicroDuckVelocityRollersFlatEnvCfg`,
which is also what upstream's own factory does -- it calls the roller factory and then edits. The
edits are the whole file: the terrain, the neutralized command, the base reset's yaw, the added
rolling entry, a rebuilt reward dict, two terminations and a single curriculum term. Everything else
-- the robot, the actuators, the sensors, the action space, both observation groups and the whole
randomization suite -- is inherited, and ``test_microduck_rollerslope_env.py`` asserts that it still
is.

Note:
    **The terrain is collided against as a heightfield, and that is load-bearing rather than a
    detail.** Against the raw triangle mesh the generator emits, this robot's tires stop carrying it
    as soon as the wheels turn -- it rides for about 0.1 s, sinks 45 mm and stops, where on the
    heightfield it rides like it does on an analytic ground plane. The sub-terrain therefore sets
    :attr:`~isaaclab_tasks.contrib.microduck.rollerslope.terrain.FlatRampTerrainCfg.convert_to_heightfield`,
    which is the representation every stock Newton-validated terrain configuration uses, and the
    conversion is lossless here because the tile is piecewise planar with no overhangs. The raw-mesh
    behaviour is a Newton contact gap worth its own report -- it reproduces at a zero-degree ramp
    angle, on stock terrain-generator output, with nothing task-specific involved -- and the control
    experiments that establish that live in
    ``artifacts/microduck/golden_trajectories/rollerslope/controls/``. It does not affect this task
    as configured.

Note:
    Upstream additionally sets ``nan_policy = "sanitize"`` on both observation groups, an mjlab
    ``ObservationGroupCfg`` field with no Isaac Lab equivalent (addendum sections 5.5 and 9.10). It
    guards a contact divergence upstream measures at roughly one step-environment in 25 million,
    where the free joint goes to NaN and reaches the observation one step before the ``nan_state``
    termination can recycle the episode. It is **not** rebuilt here: this family already carries the
    two halves that matter -- the NaN-guarded critic terms and :attr:`TerminationsCfg.nan_state` --
    and the residual exposure is the actor's own terms on that single step. Adding an actor-side
    sanitizer would mean touching every inherited observation term or the observation manager itself,
    which is a far wider change than the risk it removes.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
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
# The slope (addendum sections 2 and 3)
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
"""The sub-terrain at upstream's defaults; the generator overwrites its ``size`` with the tile's."""

MICRODUCK_SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=MICRODUCK_SLOPE_TILE_SIZE,
    curriculum=True,
    num_rows=MICRODUCK_SLOPE_NUM_ROWS,
    num_cols=1,
    difficulty_range=(0.0, 1.0),
    sub_terrains={"flat_ramp": _FLAT_RAMP_CFG},
)
"""The slope ladder MicroDuck descends (addendum section 3.1).

One sub-terrain type over ten rows of increasing difficulty, which the generator maps to the ramp
angle. The two stacks' per-row difficulty formula -- ``(row + eta) / num_rows`` rescaled into
``difficulty_range`` -- is character-for-character identical, so a row here is the same ramp band as
upstream's (addendum section 3.2).

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
"""World-frame height [m] below which the robot has left the terrain: **-3.41176 m**.

Derived rather than transcribed, exactly as upstream derives it: the deepest run-out any tile can
reach is the longest ramp at the steepest angle, ``8.0 * tan(20 deg) = 2.9118`` m below the starting
platform, and the floor sits half a metre under it. Being independent of the ramp a given
environment actually drew is the point -- it cannot fire during a nominal descent, only once the
robot is falling through the gap past a short ramp.
"""

MICRODUCK_SLOPE_FELL_OVER_LIMIT = 1.0
"""Trunk tilt [rad] that ends the episode: 57.3 degrees, tightened from the skating task's 70.

A skater that tips this far on an incline is not going to recover into a roll, and upstream ends the
episode rather than paying for the slide.
"""

MICRODUCK_SLOPE_ENTRY_SPEED_X = (0.25, 0.45)
"""Forward speed [m/s] the robot leaves the reset already rolling at (addendum section 4.3).

One draw per environment. The base *and* the four wheels are set from it, so the contact starts
rolling instead of skidding: upstream reached for this after a base-only shove produced a
first-step contact spike that diverged, and taught the policy to walk itself to a stop.
"""

MICRODUCK_SLOPE_UPRIGHT_STD = math.sqrt(0.08)
"""Width [-] of the upright reward's Gaussian, converted from upstream's rather than copied.

Upstream scores tilt with ``exp(-(1 - cos t) / 0.2^2)`` and
:func:`~isaaclab_tasks.contrib.microduck.mdp.rewards.upright_selected_body` scores
``exp(-sin^2(t) / std^2)`` (addendum section 4.6). Since ``sin^2 t = (1 - cos t)(1 + cos t)``,
matching the exponents needs ``std^2 = 0.2^2 * (1 + cos t)``, which is exact at one tilt only;
normalizing at vertical -- where the reward and its gradient matter most -- gives
``sqrt(2) * 0.2 = sqrt(0.08)``. The two agree exactly upright and ours decays slightly faster off
it: at 30 degrees upstream pays 0.0351 and this pays 0.0439, and by the 57.3 degree tilt termination
both are under 1.5e-4 -- four orders of magnitude below the vertical value, on the far side of the
angle where the episode ends.

The family's exactly-equivalent kernel, ``upright_gaussian_at_height``, carries a height gate this
task must not have, so using it would have meant a new ungated kernel for one term. That alternative
was considered and rejected (addendum section 9.6).
"""

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
    """The skating command term, kept alive and neutralized (addendum section 5.1).

    The term is **not** deleted, and that is deliberate on both sides: the 61-wide observation
    reserves three slots for the twist, and removing the term would collapse the deployed vector to
    58. So the throttle range is zeroed and every environment is made a standing one, which is what
    the base term zeroes on every step.

    The descent comes from gravity, so there is nothing to command; the skating task's throttle would
    only ask the policy to pedal on a hill it is already falling down.

    Note:
        :attr:`rel_forward_envs` is zeroed here where upstream leaves the template's 0.2, and that
        is a **deliberate deviation** rather than a tidy-up. The forced-forward bucket clamps the
        surge slot to at least 0.3 at resample time, and upstream's own
        :class:`~isaaclab_tasks.contrib.microduck.mdp.commands.RelativeHeadingVelocityCommand`
        override *replaces* the base ``_update_command`` and so never applies the
        standing-environment zeroing that ``rel_standing_envs = 1.0`` was set to trigger. On
        upstream's stack the two combine to feed a 0.3 throttle to a fifth of the environments, for
        the whole resampling interval, on a task where **no reward reads the command** -- it reaches
        only the actor's three twist slots. Turning the bucket off instead makes the twist slots
        identically zero at every step of every episode, which is what upstream's own edit was
        reaching for and what the neutralization is for here.

        Reproducing upstream's *actual* behaviour is possible -- a task-local subclass could drop
        the standing-environment zeroing without touching the class the skating and swizzle tasks
        share -- and it is rejected on cost rather than on feasibility. A new command class to
        reproduce a throttle leak into an observation slot nothing reads buys nothing; the family
        reproduces upstream accidents that shape *physics*, and this one shapes only the actor's
        observation distribution in one slot for a fifth of the environments. Note also that the
        literal one-line revert to 0.2 would reproduce **neither** stack: our shared class zeroes
        standing environments every step, so the 0.3 would survive only in the observation returned
        by the reset step, before the first update.
    """

    def __post_init__(self):
        command = self.base_velocity
        command.rel_standing_envs = 1.0
        command.rel_heading_envs = 0.0
        # see the class note: this is upstream's 0.2 turned off, not upstream's value
        command.rel_forward_envs = 0.0
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class RewardsCfg:
    """Reward terms for the MDP (addendum section 5.3).

    Nine terms, and the interesting thing about them is what is *missing*. Upstream keeps exactly one
    of the skating recipe's twenty-one -- :attr:`action_rate_l2` -- and declares six of the rest again
    with fresh parameters, so fourteen terms simply stop existing: the fixed-pose reward, the
    centre-of-mass height band, both velocity terms, the whole stroke package and the two skating
    regularizers. Upstream's own note gives the reason: *no fixed-pose reward, so it is free to bend
    and lean to hold the slope instead of being told to stand the way it does on the flat.*

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

    # Capped at a rolling speed of 0.35 m/s. Upstream leaves ``wheel_radius`` at its 0.0175 m
    # default where the model measures 0.0150 m (addendum section 9.3), which only mislabels the
    # cap: the true ground speed at saturation is 0.300 m/s. Reproduced verbatim, as everywhere else
    # this constant appears, because the reference policy trained against it.
    wheel_glide = RewTerm(
        func=mdp.wheel_glide_reward,
        weight=2.0,
        params={"asset_cfg": _WHEEL_JOINT_CFG, "cap_speed": 0.35},
    )
    # The same kernel and width the skating task uses, at 1.5 rather than 1.0: there is no throttle
    # to steer with here, so drifting off the fall line is not recoverable by pushing harder.
    heading_hold = RewTerm(func=mdp.heading_hold_reward, weight=1.5, params={"std": 0.4})
    # Upstream's own term is a constant 1; the stock one pays 0 on the step an episode ends by
    # termination rather than by time-out. One step per episode at weight 1.0 -- the same
    # substitution the rest of the family makes (addendum section 9.8).
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    ##
    # Posture on the incline.
    ##

    # The heaviest term in the dict. See :data:`MICRODUCK_SLOPE_UPRIGHT_STD` for why the width is
    # converted from upstream's 0.2 rather than copied.
    upright = RewTerm(
        func=mdp.upright_selected_body,
        weight=3.0,
        params={"std": MICRODUCK_SLOPE_UPRIGHT_STD, "asset_cfg": _TRUNK_BODY_CFG},
    )
    # The skating declaration, restated: same kernel, same weight, same per-foot contact gate and the
    # same body-axis stand-in for upstream's MJCF foot sites.
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

    # The one inherited term upstream keeps, and it is *static* here: the skating task ramps this to
    # -2.0 by iteration 500 and this task deletes that curriculum, so -1.0 is the live value
    # throughout.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    # THE ONE WEIGHT THAT BREAKS A COPY-THE-BLOCK PORT. The skating recipe charges -0.5 here and
    # upstream raises it to -0.75 for the slope, a fifty percent increase; the other three restated
    # terms are identical (addendum section 5.3). A steady head is worth more on a descent the robot
    # cannot steer out of.
    neck_joint_pos_l2 = RewTerm(func=mdp.joint_pose_l2, weight=-0.75, params={"asset_cfg": _HEAD_JOINT_CFG})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class EventsCfg(RollersEventsCfg):
    """The skating randomization suite plus the rolling entry (addendum section 5.2).

    The domain randomization is the skating task's, term for term and range for range. Two things
    change, and both are about how an episode starts.

    **Declaration order is behaviour**, since Isaac Lab fires reset events in the order they are
    declared. :attr:`reset_rolling_entry` writes the whole root velocity six-vector, so it has to run
    after ``reset_base`` places the base; inheriting appends it after every reset the skating recipe
    declares, which is where upstream appends it too.

    Note:
        Upstream also declares its ``reset_action_history`` event again here, which is a no-op
        re-registration of an inherited term (addendum section 4.9). It exists because mjlab's action
        manager does not clear the per-term history its hand-rolled action-rate penalties keep on the
        environment object. Isaac Lab's :class:`~isaaclab.managers.ActionManager` already zeroes both
        the previous and the current action for the environments being reset and then resets every
        action term, and this family's stateful action-rate penalty is a class term the reward
        manager resets on the same boundary -- so there is nothing to port and no no-op event is
        added. The one residual difference is that upstream sets the previous action to the *current*
        one where Isaac Lab sets it to zero, which is a pre-existing family-wide deviation rather
        than something this task introduces.
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
        # Face down the slope. Everything else about the base reset is the skating task's, including
        # the +/-0.5 m horizontal jitter -- which on an origin that sits on the incline can spawn the
        # trunk up to 109 mm inside the starting platform or 182 mm above the ramp at 20 degrees,
        # against a standing height of 138 mm (addendum section 9.9). That is upstream's live
        # behaviour and what the reference policy trained against, so it is reproduced rather than
        # narrowed; the accuracy-gate regimes pin it away instead.
        self.reset_base.params["pose_range"] = {**self.reset_base.params["pose_range"], "yaw": (0.0, 0.0)}
        # Restated because upstream restates it: the entry event below owns the root velocity, and a
        # reset that also injected one would be adding to a vector this overwrites.
        self.reset_base.params["velocity_range"] = {}


@configclass
class TerminationsCfg(RollersTerminationsCfg):
    """The skating terminations, tightened, plus the void guard (addendum section 5.4).

    ``time_out`` and ``nan_state`` are inherited untouched. ``fell_over`` is redeclared at a tighter
    limit angle, and one termination is added.
    """

    # The only height quantity in the whole task, and an **absolute world-z** one: it is deliberately
    # independent of the ramp a given environment drew, so it cannot fire during a nominal descent
    # and only fires once the robot has left the solid past a short ramp. Upstream reads the root
    # *link* frame where this stock term reads the root body's centre of mass; at -3.41 m the
    # centimetre between them cannot change the outcome (addendum section 9.5).
    fell_into_void = DoneTerm(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"minimum_height": MICRODUCK_SLOPE_VOID_FLOOR},
    )

    def __post_init__(self):
        self.fell_over.params["limit_angle"] = MICRODUCK_SLOPE_FELL_OVER_LIMIT


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (addendum section 5.6).

    One term, with no parameters. Upstream deletes every schedule the skating recipe declares -- the
    action-rate ramp, the wheel-friction ramp and the two centre-of-mass ramps -- and leaves only the
    terrain ladder, which has two consequences worth stating: every reward weight in
    :class:`RewardsCfg` is static for the whole run, and the wheel bearings stay at whatever the
    inherited reset event writes, which is frictionless.

    Note:
        The **first** reset of a run scores every environment against a root pose the reset events
        have not written yet -- the robot is still on the cloner's spacing grid, tens of metres from
        the terrain origin it is about to be placed at -- so the signed distance is enormous and
        every environment is promoted one rung, then demoted back on the next reset. The effect is
        one episode spent on row 1 instead of row 0. It is reproduced rather than guarded against,
        because upstream has it too: the same curriculum function, scored against the same
        row-centred grid, at the same point in the reset order.
    """

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
    """MicroDuck slope-descent environment on generated ramp terrain.

    The robot, the actuators, the sensors, the action space and both observation groups are the
    skating task's, which is what upstream arrives at as well: its factory takes the skating recipe
    and edits the terrain, the command, two reset events, the rewards, the terminations and the
    curriculum, and touches no observation term, no sensor, no randomization term, no actuator and no
    asset.
    """

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

        # Measured, not inherited, and the widest budget in the family by a factor of three. The
        # skating task's is sized for four tires on an infinite plane, which is one analytic contact
        # patch per tire; this task collides against a heightfield rasterized at 0.1 m, where a
        # sprawled robot's colliders straddle cell boundaries and pick up two triangles per cell
        # across several cells. Profiled under random actions with the tilt termination dropped and
        # the pushes forced to full magnitude, at 256, 2048 and 4096 environments: **295 constraints
        # and 92 contacts** per environment at the peak, against the skating task's 83 and 26 on the
        # same robot, and against 98 and 35 for this task on the raw mesh it no longer uses. Logs:
        # ``artifacts/microduck/profile_microduck_contacts_rollerslope_{256,2048,4096}envs.log``,
        # from ``artifacts/microduck/profile_microduck_contacts.py``.
        #
        # ``njmax`` is a hard per-environment cap and carries the wider margin; ``nconmax`` is a
        # per-environment share of one shared buffer and cannot overflow at the measured peak, so it
        # sits just above it, at the same 1.2x the rest of the family uses.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 448
        newton_mjwarp.solver_cfg.nconmax = 112
        self.sim.physics.default = newton_mjwarp
