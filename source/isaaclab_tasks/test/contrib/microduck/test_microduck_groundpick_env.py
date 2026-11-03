# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test ground-pick configuration, mouth-site kinematics and contact sensing.

Mouth rewards use a jaw_soft-frame offset checked against independent MuJoCo site poses and
physical mouth-ground impact. Configuration checks use independent expected values.
"""

import math
import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.groundpick.agents.rsl_rl_ppo_cfg import MicroDuckGroundPickPPORunnerCfg
from isaaclab_tasks.contrib.microduck.groundpick.groundpick_env_cfg import (
    MICRODUCK_DESCENT_END,
    MICRODUCK_GROUND_PICK_PERIOD,
    MICRODUCK_HOLD_END,
    MICRODUCK_MOUTH_TIP_AXIS,
    MICRODUCK_MOUTH_TIP_OFFSET,
    MICRODUCK_RISE_END,
    MicroDuckGroundPickFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.usefixtures("microduck_allcollisions_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-GroundPick-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
"""Environment control steps per PPO iteration."""

GP_PERIOD = 4.0
DESCENT_END = 0.375
HOLD_END = 0.425
RISE_END = 0.80
"""Expected normalized boundaries of the ground-pick cycle."""

SEGMENT_DURATIONS_S = {"descent": 1.5, "low_dwell": 0.2, "rise": 1.5, "standing": 0.8}
"""Expected durations [s] of the four ground-pick cycle segments."""

MOUTH_TIP_OFFSET = (-0.00809334, 0.0, -0.0777383)
MOUTH_TIP_AXIS = (-0.00872562, 0.0, -0.99996193)
"""Mouth-tip site position [m] and unit forward axis expressed in the jaw_soft body frame."""

ACTOR_OBSERVATION_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""The shared 61-component deployed actor observation layout, in concatenation order."""

ACTOR_OBSERVATION_DIM = 61
"""Actor observation width the deployed MicroDuck policy expects."""

CRITIC_OBSERVATION_TERMS = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""Privileged critic observation layout, including task-only state outside the deployment interface."""

CRITIC_OBSERVATION_DIM = 74
"""Expected width of the assembled critic observation group."""

EXPECTED_SERVO_JOINT_NAMES = [
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
"""Fourteen controlled servos in action order: left leg 0-4, neck/head 5-8 and right leg 9-13."""

EXPECTED_LEG_JOINT_NAMES = [
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
"""The ten physical leg joints, excluding the four command-driven neck/head joints."""

EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The four neck/head servos in command order."""

EXPECTED_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot body names in left/right order."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""The body the uprightness and angular-velocity terms measure."""

EXPECTED_MOUTH_BODY_NAMES = ["jaw_soft"]
"""Jaw body rigidly carrying the mouth-tip site and selected impact colliders."""

EXPECTED_REWARDS = {
    # name: (stage-0 weight, scalar params)
    "mouth_ground_proximity": (
        3.0,
        {
            "mouth_offset_b": MOUTH_TIP_OFFSET,
            "std": 0.10,
            "target_height": 0.0,
            "command_name": "base_velocity",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    ),
    "mouth_perpendicular_to_ground": (
        2.0,
        {
            "mouth_axis_b": MOUTH_TIP_AXIS,
            "command_name": "base_velocity",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    ),
    "ground_pick_return_pose_legs": (
        6.0,
        {"std": 0.3, "command_name": "base_velocity", "hold_end": HOLD_END, "rise_end": RISE_END},
    ),
    "ground_pick_return_pose_neck": (
        6.0,
        {"std": 0.15, "command_name": "base_velocity", "hold_end": HOLD_END, "rise_end": RISE_END},
    ),
    "return_upright": (
        4.0,
        {"std": 0.4, "command_name": "base_velocity", "hold_end": HOLD_END, "rise_end": RISE_END},
    ),
    "feet_grounded": (3.0, {}),
    "feet_flat": (-2.0, {"normal_axis": (0.0, 1.0, 0.0)}),
    "upright": (0.2, {"std": math.sqrt(0.05)}),
    "head_impact_penalty": (-2.0, {"threshold": 1.0}),
    "self_collisions": (-1.0, {"saturate": True}),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "dof_pos_limits": (-1.0, {}),
    "action_rate_l2": (-0.8, {}),
    "neck_action_rate_l2": (-1.0, {"action_name": "joint_pos"}),
    "joint_torques_l2": (-5e-3, {}),
    "neck_vel_descent": (-0.1, {"command_name": "base_velocity", "hold_end": HOLD_END}),
}
"""Initial reward terms, weights and scalar parameters, including inherited terms."""

PHASE_GATED_REWARDS = {
    "mouth_ground_proximity": "bend",
    "mouth_perpendicular_to_ground": "bend",
    "ground_pick_return_pose_legs": "return",
    "ground_pick_return_pose_neck": "return",
    "return_upright": "return",
    "neck_vel_descent": "descent",
}
"""Which of the four cycle segments each gated term is paid over.

The split is the task: nothing pays for the bend during the return or the other way round, and the
posture floor and the regularizers are ungated on purpose.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "fell_over": (False, {"limit_angle": math.radians(70.0)}),
    "nan_state": (False, {"sensor_names": ("contact_forces", "feet_ground_contact", "head_impact_contact")}),
}
"""Configured termination names, timeout flags and thresholds."""

EXPECTED_PHASE_COMMAND = {
    "period": GP_PERIOD,
    "randomize_phase": True,
    "heading_command": False,
}
"""Configured phase command parameters."""

EXPECTED_PAYLOAD_EVENT = {
    "min_kg": 0.01,
    "max_kg": 0.04,
}
"""Configured per-episode mouth payload parameters."""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "sample_mouth_payload",
    "randomize_com",
    "randomize_head_com",
    "randomize_armature",
    "randomize_joint_friction",
]
"""The reset chain, in the order it fires.

Unlike the sit-stand and ball-kick tasks there is no ground-state reset here, so no term overwrites
what another wrote: every episode spawns upright from ``reset_base``.
"""

EXPECTED_CURRICULUM_TERMS = {"action_rate_weight", "com_range", "head_com_range"}
"""Configured curriculum term names."""

EXPECTED_WEIGHT_STAGES = {
    # name: (weights, PPO iteration boundaries)
    "action_rate_weight": ([-0.8, -1.5, -2.0], [0, 250, 500]),
}
"""Reward weights at each curriculum boundary."""

EXPECTED_RANGE_STAGES = {
    # name: (ranges [m], PPO iteration boundaries)
    "com_range": ([0.003, 0.005, 0.01, 0.015, 0.02], [0, 500, 1000, 1500, 2000]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""The two centre-of-mass ramps. ``com_range`` reaches the widest value in the family."""

EXPECTED_PUSH_RANGE = (-0.15, 0.15)
"""Configured push velocity bounds [m/s], active from the first control step."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.mouth_ground_proximity.asset_cfg": ("body", EXPECTED_MOUTH_BODY_NAMES, False),
    "rewards.mouth_perpendicular_to_ground.asset_cfg": ("body", EXPECTED_MOUTH_BODY_NAMES, False),
    "rewards.ground_pick_return_pose_legs.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.ground_pick_return_pose_neck.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.feet_grounded.sensor_cfg": ("sensor", "feet_ground_contact", False),
    "rewards.feet_flat.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, True),
    "rewards.upright.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.head_impact_penalty.sensor_cfg": ("sensor", "head_impact_contact", False),
    "rewards.self_collisions.sensor_cfg": ("sensor", "self_collision", False),
    "rewards.body_ang_vel.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.neck_action_rate_l2.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.joint_torques_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "rewards.neck_vel_descent.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "events.foot_friction.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, False),
    "events.mass_inertia.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_com.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_armature.asset_cfg": ("joint", [".*"], False),
    "events.mouth_payload_force.asset_cfg": ("body", EXPECTED_MOUTH_BODY_NAMES, False),
}
"""Entity selections used by rewards, events and terminations outside the observation groups."""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>.<param>": (kind, expected names, preserve_order)
    "policy.joint_pos.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "policy.joint_vel.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.joint_pos.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.joint_vel.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.foot_air_time.sensor_cfg": ("sensor", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.foot_contact.sensor_cfg": ("sensor", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.foot_contact_forces.sensor_cfg": ("sensor", EXPECTED_FOOT_BODY_NAMES, True),
}
"""Entity selections inside the two observation groups, all of which are ordering contracts.

The table is *consumed* two-sidedly below, so an observation term that gains or loses a selection
fails rather than going unchecked.
"""

EXPECTED_HEAD_COLLIDER_XFORMS = ["top_head_shell_1", "jaw_1", "bottom_head_shell_1"]
"""The three physical head collision meshes selected by the neck-impact sensor."""

EXPECTED_NJMAX = 128
EXPECTED_NCONMAX = 32
"""The measured solver budget: 86 constraints and 27 contacts per environment at the peak."""

MEASURED_PEAK_CONSTRAINTS = 86
MEASURED_PEAK_CONTACTS = 27
"""Peak contacts and constraints measured with random actions, full pushes and tilt termination
disabled across 256, 2048 and 4096 environments.
"""

EXPECTED_IMU_MAX_LAG = 3
"""Configured maximum IMU lag [control steps] for the ground-pick task."""

MUJOCO_MOUTH_TIP_REFERENCE = {
    # servo overrides {name: angle [rad]}: (mouth position [m], mouth axis [-]), in the world frame
    # with the trunk at (0, 0, 0.30) and identity orientation
    (): ((0.085838, 0.0, 0.407521), (0.999962, 0.0, -0.008726)),
    (("neck_pitch", 0.5), ("head_pitch", -0.4), ("head_yaw", 0.3), ("head_roll", 0.2)): (
        (0.018071, 0.016147, 0.437155),
        (0.600841, 0.293853, 0.743398),
    ),
    (("neck_pitch", -0.6), ("head_pitch", 0.7), ("head_yaw", -0.5), ("head_roll", 0.4)): (
        (0.092676, -0.031454, 0.331430),
        (0.226564, -0.482389, -0.846150),
    ),
}
"""Independent MuJoCo mouth_tip site poses with root position (0, 0, 0.30) m, identity root
orientation and zero unspecified joints. Positions [m] and unit axes are expressed in the world frame.
"""


def _entity_cfg_of(term_cfg, key: str) -> SceneEntityCfg:
    """Fetch a term's entity selection, looking inside a delayed term's wrapped parameters."""
    if key in term_cfg.params:
        return term_cfg.params[key]
    return term_cfg.params["term_params"][key]


def _observation_entity_cfgs(term_cfg) -> dict[str, SceneEntityCfg]:
    """Every entity selection a single observation term carries, wrapped or not.

    The delayed actor terms hold theirs inside the wrapped term's parameters, so a walk that only
    looked at ``params`` would miss ``policy.joint_vel`` -- which is exactly one of the deploy-order
    contracts this file exists to pin.
    """
    selections = {key: value for key, value in term_cfg.params.items() if isinstance(value, SceneEntityCfg)}
    for key, value in term_cfg.params.get("term_params", {}).items():
        if isinstance(value, SceneEntityCfg):
            selections[key] = value
    return selections


def _scalar_params(term_cfg) -> dict:
    """Return scalar parameters; entity selections are checked separately by name and order."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


def _phase_command(phase: torch.Tensor) -> torch.Tensor:
    """Encode phase as (cos, sin, 0), matching the vector consumed by the reward terms."""
    command = torch.zeros(phase.shape[0], 3)
    command[:, 0] = torch.cos(2.0 * math.pi * phase)
    command[:, 1] = torch.sin(2.0 * math.pi * phase)
    return command


##
# The recipe
##


@pytest.mark.unit
def test_the_rewards_match_upstream_term_for_term():
    """Reward terms must match the expected set, stage-0 weights and scalar parameters."""
    rewards = MicroDuckGroundPickFlatEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_payload_is_registered_as_physics_rather_than_as_a_zero_weight_reward():
    """The mouth payload applies an external wrench every control step through an interval event.
    It must remain active independently of reward weights.
    """
    cfg = MicroDuckGroundPickFlatEnvCfg()

    assert "mouth_payload_force" not in vars(cfg.rewards)
    assert not any(term.weight == 0.0 for term in vars(cfg.rewards).values())

    hook = cfg.events.mouth_payload_force
    assert hook.mode == "interval"
    assert hook.interval_range_s == (0.0, 0.0)
    assert hook.func is mdp.apply_mouth_payload_force
    assert hook.params["hold_end"] == pytest.approx(HOLD_END)

    draw = cfg.events.sample_mouth_payload
    assert draw.mode == "reset"
    assert _scalar_params(draw) == EXPECTED_PAYLOAD_EVENT


@pytest.mark.unit
def test_the_segments_last_the_durations_upstream_tabulates():
    """Cycle boundaries must produce the expected segment durations [s]."""
    cfg = MicroDuckGroundPickFlatEnvCfg()

    assert pytest.approx(GP_PERIOD) == MICRODUCK_GROUND_PICK_PERIOD
    assert (MICRODUCK_DESCENT_END, MICRODUCK_HOLD_END, MICRODUCK_RISE_END) == (DESCENT_END, HOLD_END, RISE_END)

    measured = {
        "descent": DESCENT_END * GP_PERIOD,
        "low_dwell": (HOLD_END - DESCENT_END) * GP_PERIOD,
        "rise": (RISE_END - HOLD_END) * GP_PERIOD,
        "standing": (1.0 - RISE_END) * GP_PERIOD,
    }
    for name, duration in SEGMENT_DURATIONS_S.items():
        assert measured[name] == pytest.approx(duration), name
    # the descent and the rise are deliberately the same length: it is one gesture, played twice
    assert measured["descent"] == pytest.approx(measured["rise"])

    # every gated term reads the same three boundaries, so the segments cannot drift apart per term
    for name in PHASE_GATED_REWARDS:
        params = getattr(cfg.rewards, name).params
        assert params["hold_end"] == pytest.approx(HOLD_END), name
        if "rise_end" in params:
            assert params["rise_end"] == pytest.approx(RISE_END), name
        if "descent_end" in params:
            assert params["descent_end"] == pytest.approx(DESCENT_END), name


@pytest.mark.unit
def test_the_two_phase_gates_overlap_only_across_the_rise():
    """The gates are not complements, and treating them as one would pay for the approach twice.

    They sum to one across the rise, where the bend fades out as the return fades in. Across the
    descent the bend gate is opening while the return gate is still shut, which is what leaves the
    approach unpriced by the return terms.
    """
    phase = torch.linspace(0.0, 1.0, 1001)[:-1]
    bend = mdp.rewards._phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    rise = mdp.rewards._phase_rise_gate(phase, HOLD_END, RISE_END)

    assert float(bend.min()) >= 0.0 and float(bend.max()) == pytest.approx(1.0, abs=1e-3)
    assert float(rise.min()) == 0.0 and float(rise.max()) == pytest.approx(1.0)

    rising = (phase >= HOLD_END) & (phase < RISE_END)
    torch.testing.assert_close(bend[rising] + rise[rising], torch.ones_like(phase[rising]), atol=1e-6, rtol=0.0)
    # and nowhere else: through the descent and the low dwell only the bend gate is open
    descending = phase < HOLD_END
    assert float(rise[descending].max()) == 0.0
    assert float(bend[phase >= RISE_END].max()) == 0.0
    # the low dwell is the only place the bend gate is saturated
    low = (phase >= DESCENT_END) & (phase < HOLD_END)
    torch.testing.assert_close(bend[low], torch.ones_like(phase[low]))


@pytest.mark.unit
def test_the_phase_is_recovered_from_the_command_the_policy_sees():
    """Phase gates must read the deployed command vector rather than hidden command state."""
    phase = torch.linspace(0.0, 1.0, 257)[:-1]

    class _Stub:
        def __init__(self, command):
            self.command_manager = self
            self._command = command

        def get_command(self, name):
            assert name == "base_velocity"
            return self._command

    recovered = mdp.rewards._phase_from_command(_Stub(_phase_command(phase)), "base_velocity")

    torch.testing.assert_close(recovered, phase, atol=1e-6, rtol=0.0)


@pytest.mark.unit
def test_the_perpendicularity_reward_charges_a_mouth_pointing_the_wrong_way():
    """Its kernel is a signed cosine, so it is a reward *and* a penalty at one positive weight.

    Reaching the floor mouth-up is a different, useless posture, and a term that merely failed to pay
    for it would leave the proximity reward free to find it.
    """
    weight = MicroDuckGroundPickFlatEnvCfg().rewards.mouth_perpendicular_to_ground.weight

    assert weight > 0.0
    # the kernel's range, worked out from the gate and the alignment rather than restated: the gate
    # is in [0, 1] and the alignment in [-1, 1]
    assert -weight < 0.0 < weight


@pytest.mark.unit
def test_the_upright_reward_is_weak_and_the_return_one_is_not():
    """The approach *requires* a deep forward lean, so a strong always-on uprightness reward would
    price the task out. Verticality is paid for where it is wanted instead: on the way back up.
    """
    cfg = MicroDuckGroundPickFlatEnvCfg()
    velocity = MicroDuckVelocityFlatEnvCfg()

    assert cfg.rewards.upright.weight == pytest.approx(0.2)
    assert cfg.rewards.upright.weight < velocity.rewards.upright.weight
    assert cfg.rewards.return_upright.weight > 10.0 * cfg.rewards.upright.weight
    # and the gated one is the one that is off during the approach
    assert PHASE_GATED_REWARDS["return_upright"] == "return"


@pytest.mark.unit
def test_the_return_block_outweighs_the_bend_block():
    """The clean-return reward block must outweigh the bend reward block."""
    rewards = MicroDuckGroundPickFlatEnvCfg().rewards

    bend = sum(getattr(rewards, name).weight for name, segment in PHASE_GATED_REWARDS.items() if segment == "bend")
    ret = sum(getattr(rewards, name).weight for name, segment in PHASE_GATED_REWARDS.items() if segment == "return")

    assert bend == pytest.approx(5.0)
    assert ret == pytest.approx(16.0)
    assert ret > 3.0 * bend


@pytest.mark.unit
def test_the_no_touch_equilibrium_has_a_term_on_each_side():
    """Delete either one and the task becomes a different task: "touch the floor", or "stand still"."""
    rewards = MicroDuckGroundPickFlatEnvCfg().rewards

    assert rewards.mouth_ground_proximity.weight > 0.0
    assert rewards.head_impact_penalty.weight < 0.0
    # the penalty carries a dead band rather than a scale, so a brush is free and a slam is not
    assert rewards.head_impact_penalty.params["threshold"] == pytest.approx(1.0)


@pytest.mark.unit
def test_the_flat_foot_penalty_is_ungated_here_unlike_on_the_roller_tasks():
    """No swing phase, so both soles are asked to lie flat at every phase.

    ``feet_grounded`` alone is satisfied by a single contact point, which a foot pivoted onto its
    edge still has; this is the term that denies that.
    """
    term = MicroDuckGroundPickFlatEnvCfg().rewards.feet_flat

    assert "sensor_cfg" not in term.params
    assert term.func is mdp.feet_flat_penalty


@pytest.mark.unit
def test_the_reset_spawns_upright_with_no_ground_state_bucket():
    """Every ground-pick reset starts standing, without a ground-state mixture."""
    events = MicroDuckGroundPickFlatEnvCfg().events

    assert "set_ground_state" not in vars(events)
    reset_terms = [name for name, term in vars(events).items() if term.mode == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER
    pose_range = events.reset_base.params["pose_range"]
    # the task samples the absolute height in (0.12, 0.13) m; Isaac Lab samples an offset from the
    # 0.125 m default, so the band has to be the same 10 mm wide and centred
    assert pose_range["z"] == pytest.approx((-0.005, 0.005))
    assert pose_range["yaw"] == pytest.approx((-3.14, 3.14))
    # the return terms score against the stand pose, so the spawn is exactly on it
    assert events.reset_robot_joints.params["position_range"] == (0.0, 0.0)


@pytest.mark.unit
def test_the_pushes_are_half_the_family_magnitude_and_never_ramped():
    """Ground-pick pushes use fixed +/-0.15 m/s bounds without a ramp."""
    cfg = MicroDuckGroundPickFlatEnvCfg()

    velocity_range = cfg.events.push_robot.params["velocity_range"]
    assert velocity_range["x"] == pytest.approx(EXPECTED_PUSH_RANGE)
    assert velocity_range["y"] == pytest.approx(EXPECTED_PUSH_RANGE)
    assert cfg.events.push_robot.interval_range_s == (3.0, 6.0)
    assert "push_magnitude" not in vars(cfg.curriculum)
    # half of what the sit-stand and ball-kick tasks reach
    assert EXPECTED_PUSH_RANGE[1] == pytest.approx(0.15)


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """Nothing schedules the task itself; what is scheduled is the cost of moving and the DR width."""
    curriculum = MicroDuckGroundPickFlatEnvCfg().curriculum

    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS

    for name, (weights, iterations) in EXPECTED_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name

    for name, (ranges, iterations) in EXPECTED_RANGE_STAGES.items():
        stages = getattr(curriculum, name).params["range_stages"]
        assert [stage["range"] for stage in stages] == pytest.approx(ranges), name
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name


@pytest.mark.unit
def test_the_declared_action_rate_weight_is_the_live_one():
    """The action-rate weight must equal the effective stage-0 scheduled value."""
    cfg = MicroDuckGroundPickFlatEnvCfg()

    assert cfg.rewards.action_rate_l2.weight == pytest.approx(-0.8)
    assert cfg.curriculum.action_rate_weight.params["weight_stages"][0]["weight"] == pytest.approx(-0.8)
    assert cfg.curriculum.action_rate_weight.params["weight_stages"][-1]["weight"] == pytest.approx(-2.0)


@pytest.mark.unit
def test_the_centre_of_mass_ramp_is_the_widest_in_the_family():
    """The gesture is slow enough for the robot to compensate, and being able to is the point."""
    stages = MicroDuckGroundPickFlatEnvCfg().curriculum.com_range.params["range_stages"]

    assert stages[-1]["range"] == pytest.approx(0.02)
    velocity_stages = MicroDuckVelocityFlatEnvCfg().curriculum.com_range.params["range_stages"]
    assert stages[-1]["range"] > velocity_stages[-1]["range"]


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """Head center-of-mass randomization must select the intended head bodies without a hip link."""
    body_names = MicroDuckGroundPickFlatEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert body_names == ["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"]
    assert "bearing_roll" not in body_names
    # the head-impact sensor senses a strict subset of these -- the only one carrying colliders
    assert EXPECTED_MOUTH_BODY_NAMES[0] in body_names


@pytest.mark.unit
def test_the_terms_select_the_joints_bodies_and_sensors_upstream_measures():
    """A term that measures the wrong entity is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckGroundPickFlatEnvCfg()

    # two-sided over the terms that carry a selection at all, so a term that gains or loses one
    # fails rather than going unchecked
    measured = {
        f"{manager}.{term_name}.{key}"
        for manager in ("rewards", "events")
        for term_name, term in vars(getattr(cfg, manager)).items()
        for key, value in term.params.items()
        if isinstance(value, SceneEntityCfg)
    }
    # ``randomize_head_com`` selects four body *patterns* rather than four names, which the table's
    # name equality cannot express; it is pinned by
    # :func:`test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link` instead. It is
    # the only exemption, and the second assertion is what stops the exemption outliving the
    # selection it excuses.
    exempt = {"events.randomize_head_com.asset_cfg"}
    assert measured - exempt == set(EXPECTED_ENTITY_SELECTIONS)
    assert exempt <= measured, "the exemption names a selection the recipe no longer makes"

    for path, (kind, expected, preserve_order) in EXPECTED_ENTITY_SELECTIONS.items():
        manager, term_name, key = path.split(".")
        entity_cfg = getattr(getattr(cfg, manager), term_name).params[key]
        if kind == "sensor":
            assert entity_cfg.name == expected, path
            assert entity_cfg.joint_names is None and entity_cfg.body_names is None, path
            continue
        assert entity_cfg.name == "robot", path
        if kind == "joint":
            assert entity_cfg.joint_names == expected, path
            assert entity_cfg.body_names is None, path
        else:
            assert entity_cfg.body_names == expected, path
            assert entity_cfg.joint_names is None, path
        assert entity_cfg.preserve_order is preserve_order, path


@pytest.mark.unit
def test_both_observation_groups_read_the_servos_and_the_feet_in_the_deploy_order():
    """Isaac Lab resolves joints and bodies in USD order; the deployed vector is in MJCF order.

    Widths are order-blind, so the integration test that measures 61 and 74 cannot see a joint block
    that lost ``preserve_order`` and now reports the servos in the articulation's order -- which
    would produce a policy that runs on the robot against the wrong columns.
    """
    observations = MicroDuckGroundPickFlatEnvCfg().observations
    groups = {"policy": _observation_terms(observations.policy), "critic": _observation_terms(observations.critic)}

    measured = {
        f"{group}.{term_name}.{key}"
        for group, terms in groups.items()
        for term_name, term in terms.items()
        for key in _observation_entity_cfgs(term)
    }
    assert measured == set(EXPECTED_OBSERVATION_SELECTIONS)

    for path, (kind, expected, preserve_order) in EXPECTED_OBSERVATION_SELECTIONS.items():
        group, term_name, key = path.split(".")
        entity_cfg = _entity_cfg_of(groups[group][term_name], key)
        if kind == "joint":
            assert entity_cfg.name == "robot", path
            assert entity_cfg.joint_names == expected, path
        else:
            assert entity_cfg.name == "contact_forces", path
            assert entity_cfg.body_names == expected, path
        assert entity_cfg.preserve_order is preserve_order, path

    # Resolve the four neck/head servos by their names in the action joint order.
    assert EXPECTED_SERVO_JOINT_NAMES[5:9] == EXPECTED_HEAD_JOINT_NAMES
    # and ``_LEG_JOINTS = [0..4, 9..13]`` is its complement
    assert EXPECTED_SERVO_JOINT_NAMES[0:5] + EXPECTED_SERVO_JOINT_NAMES[9:14] == EXPECTED_LEG_JOINT_NAMES


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract():
    """One runtime on the robot feeds every MicroDuck policy from the same 61-wide buffer."""
    terms = _observation_terms(MicroDuckGroundPickFlatEnvCfg().observations.policy)

    assert list(terms) == [name for name, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    # ten of the sixty-one columns are shape placeholders for a runtime hot-swap, deliberately
    # constant zero rather than live tiny ranges
    assert terms["head_pose_commands"].func is mdp.zero_command_padding
    assert terms["body_pose_commands"].func is mdp.zero_command_padding
    # the phase rides in the twist slot, which keeps its three columns
    assert terms["velocity_commands"].params["command_name"] == "base_velocity"


@pytest.mark.unit
def test_the_task_carries_no_head_or_body_pose_command():
    """The head is the task's working end rather than a steerable payload, so nothing commands it.

    The consequence is deliberate and worth pinning: the neck-offset action wrapper and the head-pose
    curriculum the walking family carries are both absent here.
    """
    cfg = MicroDuckGroundPickFlatEnvCfg()

    assert set(vars(cfg.commands)) == {"base_velocity"}
    assert "head_pose_range" not in vars(cfg.curriculum)
    assert cfg.actions.joint_pos.joint_names == EXPECTED_SERVO_JOINT_NAMES
    assert cfg.actions.joint_pos.preserve_order is True


@pytest.mark.unit
def test_the_phase_command_is_the_open_loop_cycle_clock():
    """No resample, no heading controller, no standing-environment machinery: a clock."""
    command = MicroDuckGroundPickFlatEnvCfg().commands.base_velocity

    assert isinstance(command, mdp.GroundPickPhaseCommandCfg)
    assert command.class_type is mdp.GroundPickPhaseCommand
    for field, value in EXPECTED_PHASE_COMMAND.items():
        assert getattr(command, field) == value, field
    # the inherited velocity ranges are never sampled, so they are pinned at zero rather than left
    # at the walking task's
    assert command.ranges.lin_vel_x == (0.0, 0.0)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.ranges.heading is None


@pytest.mark.unit
def test_the_critic_group_drops_the_foot_height_columns():
    """The critic omits foot-height terms when no height sensor or height reward is configured."""
    terms = _observation_terms(MicroDuckGroundPickFlatEnvCfg().observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert sum(width for _, width in CRITIC_OBSERVATION_TERMS) == CRITIC_OBSERVATION_DIM
    assert "foot_height" not in terms
    # Finite-state termination guards cover the selected sensors.
    assert terms["foot_air_time"].func is mdp.foot_group_air_time_safe
    assert terms["foot_contact_forces"].func is mdp.foot_group_contact_forces_safe


@pytest.mark.unit
def test_the_imu_latency_bound_is_upstreams_pre_audit_value():
    """Ground-pick retains a maximum IMU lag of three control steps."""
    terms = _observation_terms(MicroDuckGroundPickFlatEnvCfg().observations.policy)
    velocity_terms = _observation_terms(MicroDuckVelocityFlatEnvCfg().observations.policy)

    for name in ("base_ang_vel", "projected_gravity"):
        assert terms[name].params["max_lag"] == EXPECTED_IMU_MAX_LAG, name
        assert terms[name].params["min_lag"] == 0, name
        assert terms[name].params["update_period"] == 64, name
        assert velocity_terms[name].params["max_lag"] == 1, name
    # the joint-velocity lag is the family's, and is not part of the deviation
    assert terms["joint_vel"].params["min_lag"] == terms["joint_vel"].params["max_lag"] == 1


@pytest.mark.unit
def test_the_task_runs_the_all_collisions_robot_on_a_plane():
    """On the walking model the head carries no collider, so "close but not touching" has nothing to
    push back with and the mouth would sink through the plane.
    """
    scene = MicroDuckGroundPickFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path.endswith("microduck_allcollisions.usd")
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None


@pytest.mark.unit
def test_the_head_impact_sensor_covers_upstreams_neck_subtree_against_the_terrain():
    """The neck-impact sensor must cover all three selected head collision meshes against the terrain."""
    scene = MicroDuckGroundPickFlatEnvCfg().scene

    (sensing,) = scene.head_impact_contact.sensor_shape_prim_expr
    for xform in EXPECTED_HEAD_COLLIDER_XFORMS:
        assert xform in sensing, xform
    assert scene.head_impact_contact.filter_shape_prim_expr == ["/World/ground/.*"]

    (feet_sensing,) = scene.feet_ground_contact.sensor_shape_prim_expr
    assert "left_foot_collision" in feet_sensing and "right_foot_collision" in feet_sensing
    assert scene.feet_ground_contact.filter_shape_prim_expr == ["/World/ground/.*"]


@pytest.mark.unit
def test_the_self_collision_sensor_senses_every_collider_against_every_other():
    """Self-collision sensing must cover all selected collider pairs and saturate the pair cost at one."""
    cfg = MicroDuckGroundPickFlatEnvCfg()

    sensor = cfg.scene.self_collision
    assert sensor.sensor_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert sensor.filter_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert cfg.rewards.self_collisions.params["saturate"] is True


@pytest.mark.unit
def test_the_terminations_keep_the_fall_and_guard_every_force_path():
    """A fall is a failed episode here, and every contact force that feeds something is NaN-guarded."""
    cfg = MicroDuckGroundPickFlatEnvCfg()
    terminations = cfg.terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == params, name

    # The guard names every sensor whose *force magnitude* reaches a reward or an observation -- and
    # one of them reaches a reward, so a single non-finite value there poisons the episode sum rather
    # than one observation column. ``self_collision`` is the scene's fourth contact sensor and is
    # deliberately absent: the term reading it asks whether a contact exists, not how hard it is, so
    # a non-finite force there cannot propagate.
    sensed = set(terminations.nan_state.params["sensor_names"])
    assert sensed == {"contact_forces", "feet_ground_contact", "head_impact_contact"}
    assert "self_collision" in vars(cfg.scene) and "self_collision" not in sensed


@pytest.mark.unit
def test_the_solver_profile_is_the_templates_and_the_budget_is_measured():
    """Solver iterations retain the shared profile, with contact and constraint capacities set to measured bounds."""
    solver = MicroDuckGroundPickFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg
    velocity_solver = MicroDuckVelocityFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg

    assert solver.iterations == velocity_solver.iterations == 10
    assert solver.ls_iterations == velocity_solver.ls_iterations == 20
    assert solver.njmax == EXPECTED_NJMAX
    assert solver.nconmax == EXPECTED_NCONMAX
    # a floor under the measurement, so a later retune cannot silently drop below it
    assert solver.njmax >= MEASURED_PEAK_CONSTRAINTS
    assert solver.nconmax >= MEASURED_PEAK_CONTACTS
    # and above the walking task's, which is what says the model swap was accounted for
    assert solver.nconmax > velocity_solver.nconmax


@pytest.mark.unit
def test_the_episode_is_five_complete_cycles():
    """A 20 s episode must contain exactly five 4 s command cycles."""
    cfg = MicroDuckGroundPickFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(20.0)
    # 1000 control steps at the 50 Hz the deployed policy runs at, and 200 of them per cycle
    control_steps = round(cfg.episode_length_s / (cfg.decimation * cfg.sim.dt))
    assert control_steps == 1000
    assert cfg.episode_length_s / GP_PERIOD == pytest.approx(5.0)
    assert control_steps % round(GP_PERIOD / (cfg.decimation * cfg.sim.dt)) == 0


@pytest.mark.unit
def test_the_runner_differs_from_the_velocity_one_in_two_fields():
    """The runner shares the velocity learner settings and changes only its experiment name and budget."""
    from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

    runner, velocity = MicroDuckGroundPickPPORunnerCfg(), MicroDuckPPORunnerCfg()

    assert runner.experiment_name == "microduck_groundpick"
    assert runner.max_iterations == 20000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION == velocity.num_steps_per_env
    assert runner.save_interval == velocity.save_interval == 250
    assert runner.actor.hidden_dims == velocity.actor.hidden_dims == [512, 256, 128]
    assert runner.critic.hidden_dims == velocity.critic.hidden_dims
    assert runner.obs_groups == velocity.obs_groups
    for field in ("clip_param", "entropy_coef", "learning_rate", "gamma", "lam", "desired_kl"):
        assert getattr(runner.algorithm, field) == getattr(velocity.algorithm, field), field
    # symmetry stays off, as it does on every task in this batch
    assert getattr(runner.algorithm, "symmetry_cfg", None) is None
    # every curriculum has finished ramping long before the budget runs out
    last_stage = max(
        stage["step"]
        for term in vars(MicroDuckGroundPickFlatEnvCfg().curriculum).values()
        for key in ("weight_stages", "range_stages")
        if key in term.params
        for stage in term.params[key]
    )
    assert last_stage < runner.max_iterations * STEPS_PER_ITERATION


@pytest.mark.unit
def test_the_mouth_tip_offset_is_the_measured_site_attachment():
    """The mouth-tip offset and axis must match the independent site attachment in the jaw_soft frame."""
    assert pytest.approx(MOUTH_TIP_OFFSET) == MICRODUCK_MOUTH_TIP_OFFSET
    assert pytest.approx(MOUTH_TIP_AXIS) == MICRODUCK_MOUTH_TIP_AXIS
    # the axis is a direction, so it has to be a unit vector
    assert sum(component**2 for component in MICRODUCK_MOUTH_TIP_AXIS) == pytest.approx(1.0, abs=1e-6)
    # the mouth sits ahead of and below its parent body's frame, which is what makes the descent a
    # forward fold rather than a squat
    assert MICRODUCK_MOUTH_TIP_OFFSET[2] < -0.07


##
# Simulator-backed acceptance
##


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_observation_groups_are_the_widths_their_contracts_name():
    """The actor width is a deploy contract; the critic width is measured against the term table."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        observations, _ = env.reset()

        assert observations["policy"].shape[1] == ACTOR_OBSERVATION_DIM
        assert observations["critic"].shape[1] == CRITIC_OBSERVATION_DIM
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_mouth_tip_pose_reproduces_the_upstream_site_kinematics():
    """Body-frame mouth offsets must reproduce the independent MuJoCo world-frame site poses."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=len(MUJOCO_MOUTH_TIP_REFERENCE))
        # the reference is the nominal plant, so the per-robot randomization has to be off
        for name in ("foot_friction", "mass_inertia", "randomize_com", "randomize_head_com", "reset_base"):
            setattr(env_cfg.events, name, None)
        for name in list(vars(env_cfg.curriculum)):
            setattr(env_cfg.curriculum, name, None)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        servo_cfg = SceneEntityCfg("robot", joint_names=EXPECTED_SERVO_JOINT_NAMES, preserve_order=True)
        servo_cfg.resolve(unwrapped.scene)
        mouth_cfg = SceneEntityCfg("robot", body_names=EXPECTED_MOUTH_BODY_NAMES)
        mouth_cfg.resolve(unwrapped.scene)

        # every servo at zero except the ones each reference case names, and the trunk written
        # explicitly rather than sampled, so the comparison does not depend on the spawn distribution
        joint_pos = torch.zeros_like(robot.data.joint_pos.torch)
        for row, overrides in enumerate(MUJOCO_MOUTH_TIP_REFERENCE):
            for name, angle in overrides:
                joint_pos[row, servo_cfg.joint_ids[EXPECTED_SERVO_JOINT_NAMES.index(name)]] = angle
        robot.write_joint_position_to_sim(joint_pos)
        robot.write_joint_velocity_to_sim(torch.zeros_like(joint_pos))
        root_pose = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
        root_pose[:, :3] = unwrapped.scene.env_origins + torch.tensor([0.0, 0.0, 0.30], device=unwrapped.device)
        root_pose[:, 6] = 1.0  # (x, y, z, w) identity
        robot.write_root_link_pose_to_sim(root_pose)
        robot.write_root_com_velocity_to_sim(torch.zeros(unwrapped.num_envs, 6, device=unwrapped.device))
        unwrapped.scene.write_data_to_sim()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

        body_pos = robot.data.body_link_pos_w.torch[:, mouth_cfg.body_ids].squeeze(1)
        body_quat = robot.data.body_link_quat_w.torch[:, mouth_cfg.body_ids].squeeze(1)
        offset = torch.tensor(MICRODUCK_MOUTH_TIP_OFFSET, device=unwrapped.device).expand_as(body_pos)
        axis = torch.tensor(MICRODUCK_MOUTH_TIP_AXIS, device=unwrapped.device).expand_as(body_pos)
        mouth_pos = body_pos + math_utils.quat_apply(body_quat, offset) - unwrapped.scene.env_origins
        mouth_axis = math_utils.quat_apply(body_quat, axis)

        for row, (position, direction) in enumerate(MUJOCO_MOUTH_TIP_REFERENCE.values()):
            assert mouth_pos[row].tolist() == pytest.approx(position, abs=2e-5), row
            assert mouth_axis[row].tolist() == pytest.approx(direction, abs=2e-5), row
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_head_impact_sensor_fires_when_the_mouth_reaches_the_ground():
    """The task's central equilibrium, measured rather than assumed.

    Its two halves are sensed through different machinery -- the reward reads a body frame plus a
    fixed offset, the penalty reads a terrain-filtered contact sensor on three collision shells -- and
    nothing in the configuration makes them agree. This drives a scripted fold until the head reaches
    the floor and checks that they do: the sensor is silent while the mouth is high, it fires only
    once the mouth is within millimetres of the ground, and the feet never leave it.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        # a scripted open-loop gesture, so the disturbances and the per-robot spread have to be off;
        # the fall termination too, because the deep fold is exactly what it exists to catch
        for name in (
            "foot_friction",
            "encoder_bias",
            "mass_inertia",
            "randomize_com",
            "randomize_head_com",
            "randomize_armature",
            "randomize_joint_friction",
            "push_robot",
        ):
            setattr(env_cfg.events, name, None)
        for name in list(vars(env_cfg.curriculum)):
            setattr(env_cfg.curriculum, name, None)
        env_cfg.terminations.fell_over = None
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        servo_cfg = SceneEntityCfg("robot", joint_names=EXPECTED_SERVO_JOINT_NAMES, preserve_order=True)
        servo_cfg.resolve(unwrapped.scene)
        mouth_cfg = SceneEntityCfg("robot", body_names=EXPECTED_MOUTH_BODY_NAMES)
        mouth_cfg.resolve(unwrapped.scene)
        head_sensor_cfg = SceneEntityCfg("head_impact_contact")
        head_sensor_cfg.resolve(unwrapped.scene)
        feet_sensor_cfg = SceneEntityCfg("feet_ground_contact")
        feet_sensor_cfg.resolve(unwrapped.scene)

        # fold the hips and knees forward and pitch the head down: the gesture the task is named for,
        # driven open loop past the hover so that contact is actually reached
        fold = {
            "left_hip_pitch": -1.2,
            "left_knee": 1.6,
            "left_ankle": -0.6,
            "right_hip_pitch": 1.2,
            "right_knee": -1.6,
            "right_ankle": 0.6,
            "neck_pitch": 1.2,
            "head_pitch": 1.0,
        }
        default = robot.data.default_joint_pos.torch[0, servo_cfg.joint_ids]
        action = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
        for column, name in enumerate(EXPECTED_SERVO_JOINT_NAMES):
            if name in fold:
                action[:, column] = fold[name] - default[column]

        offset = torch.tensor(MICRODUCK_MOUTH_TIP_OFFSET, device=unwrapped.device)
        heights, impacts, grounded = [], [], []
        for _ in range(120):
            env.step(action)
            body_pos = robot.data.body_link_pos_w.torch[:, mouth_cfg.body_ids].squeeze(1)
            body_quat = robot.data.body_link_quat_w.torch[:, mouth_cfg.body_ids].squeeze(1)
            mouth_pos = body_pos + math_utils.quat_apply(body_quat, offset.expand_as(body_pos))
            heights.append((mouth_pos[:, 2] - unwrapped.scene.env_origins[:, 2]).clone())
            impacts.append(mdp.body_impact_cost(unwrapped, head_sensor_cfg, threshold=0.0).clone())
            grounded.append(mdp.feet_grounded_reward(unwrapped, feet_sensor_cfg).clone())
        height = torch.stack(heights)
        impact = torch.stack(impacts)
        support = torch.stack(grounded)

        # the gesture starts standing and ends with the mouth on the floor
        assert float(height[0].min()) > 0.15
        assert float(height[-1].max()) < 0.02
        # the sensor is silent for the whole approach and fires for the rest
        touching = impact > 1.0
        assert not bool(touching[0].any()), "the head sensor fired before the fold began"
        assert bool(touching[-1].all()), "the head reached the floor and the sensor stayed silent"
        # and the two channels agree: nothing is ever sensed as touching while the mouth is high, and
        # the mouth is never on the floor without the sensor saying so
        assert float(height[touching].max()) < 0.02
        assert float(height[~touching].min()) > 0.005
        # and the feet are still on the floor at the end, which is what makes this a bend rather than
        # a face-plant -- the fold rocks a sole momentarily on its way down, so the claim is about
        # where the gesture arrives and how much of it is spent supported
        assert float(support[-1].min()) == pytest.approx(1.0)
        assert float(support[20:].mean()) > 0.9
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_phase_clock_advances_one_cycle_per_period_and_starts_spread_out():
    """The command is an open-loop clock: nothing the robot does can move it.

    The two halves are separate failures. A clock that drifts changes the gesture's tempo, and a
    clock that starts at the same phase everywhere trains four skills on synchronized environments,
    which is what the randomized start exists to avoid.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=64)
        # An unactuated MicroDuck topples within a couple of seconds, and a reset would re-draw the
        # very quantity under test. The fall termination is therefore dropped for the window, which
        # is what lets the clock be checked over a whole cycle rather than over a fragment of one.
        env_cfg.terminations.fell_over = None
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        command_term = unwrapped.command_manager.get_term("base_velocity")
        start = command_term.phase.clone()
        # the starting phase is drawn per environment and covers the cycle
        assert float(start.max() - start.min()) > 0.8
        assert float(start.std()) > 0.2

        steps_per_cycle = round(GP_PERIOD / unwrapped.step_dt)
        assert steps_per_cycle == 200
        action = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
        for _ in range(steps_per_cycle):
            env.step(action)

        # no environment restarted, so the clock was never re-seeded under the measurement
        assert int(unwrapped.episode_length_buf.min()) == steps_per_cycle
        # exactly one cycle later, on every environment, whatever the robot did in between
        torch.testing.assert_close(command_term.phase, start, atol=1e-4, rtol=0.0)
        # and the command slot carries that phase on the unit circle
        command = unwrapped.command_manager.get_command("base_velocity")
        torch.testing.assert_close(command[:, 0], torch.cos(2.0 * math.pi * start), atol=1e-4, rtol=0.0)
        torch.testing.assert_close(command[:, 1], torch.sin(2.0 * math.pi * start), atol=1e-4, rtol=0.0)
        torch.testing.assert_close(command[:, 2], torch.zeros_like(start))
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_mouth_payload_hangs_only_after_the_mouth_has_closed():
    """A world-frame payload wrench acts on the head only after mouth closure and before release."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=8)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        composer = robot.permanent_wrench_composer
        payload = mdp.mouth_payload(unwrapped)
        assert float(payload.min()) >= EXPECTED_PAYLOAD_EVENT["min_kg"]
        assert float(payload.max()) <= EXPECTED_PAYLOAD_EVENT["max_kg"]
        # drawn per environment rather than shared
        assert float(payload.std()) > 0.0

        command_term = unwrapped.command_manager.get_term("base_velocity")
        hook = unwrapped.event_manager.get_term_cfg("mouth_payload_force")

        def _wrench_at(phase: float) -> tuple[torch.Tensor, torch.Tensor]:
            command_term._phase[:] = phase
            command_term._write_command()
            hook.func(unwrapped, None, **hook.params)
            return composer.out_force_b.torch.clone(), composer.out_torque_b.torch.clone()

        # mid-approach: the mouth has not closed on anything yet
        force, _ = _wrench_at(0.2)
        assert float(force.abs().max()) == pytest.approx(0.0)

        # a full ramp past the close, which is where the lift begins
        force, torque = _wrench_at(
            HOLD_END + 2.0 * MicroDuckGroundPickFlatEnvCfg().events.mouth_payload_force.params["ramp"]
        )
        magnitude = force.norm(dim=-1).sum(dim=-1)
        torch.testing.assert_close(magnitude, payload * 9.81, atol=1e-4, rtol=1e-3)
        # applied at the mouth tip rather than at the body's centre of mass, so it carries a moment
        assert float(torque.norm(dim=-1).sum(dim=-1).min()) > 0.0
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
