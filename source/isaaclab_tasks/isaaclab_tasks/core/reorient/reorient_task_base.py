# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural definitions shared by the reorientation task family.

Joint/body name lists, marker geometry, and reset-pose offsets consumed by both
the Direct and manager-based variants, plus the manager term factories and the
shared termination section used by the robot-specific manager tasks. The
factory defaults carry the Shadow Hand state-task values; each variant declares
its section class stating only its deltas.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp

GOAL_MARKER_POSITION: tuple[float, float, float] = (-0.2, -0.45, 0.68)
"""Fixed goal-marker display position [m], environment frame (state-based tasks)."""

CAMERA_GOAL_MARKER_POSITION: tuple[float, float, float] = (-0.2, 0.1, 0.6)
"""Goal-marker display position [m] for the camera tasks.

Deviates from :data:`GOAL_MARKER_POSITION` so the goal cube sits inside the
tiled camera's frustum.
"""

IN_HAND_POS_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.04)
"""Offset from the object's default position to the in-hand goal anchor [m].

Defines the Direct/manager goal-position parity: the Direct environments and
the manager command terms derive the same in-hand target point from the
object's default root position plus this offset.
"""

CAMERA_PLAY_NUM_ENVS: int = 64
"""Camera-task environment count for checkpoint playback."""

SHADOW_FINGERTIP_BODY_NAMES: list[str] = [
    "robot0_ffdistal",
    "robot0_mfdistal",
    "robot0_rfdistal",
    "robot0_lfdistal",
    "robot0_thdistal",
]
"""Shadow Hand fingertip body names (identical on every backend asset)."""

ALLEGRO_ACTUATED_JOINT_NAMES: list[str] = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]
"""Allegro Hand actuated joint names, in the Direct task's actuation order."""

ALLEGRO_FINGERTIP_BODY_NAMES: list[str] = [
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
    "thumb_link_3",
]
"""Allegro Hand fingertip body names."""

SHADOW_ACTUATED_JOINT_NAMES: list[str] = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ3",
    "robot0_FFJ2",
    "robot0_FFJ1",
    "robot0_MFJ3",
    "robot0_MFJ2",
    "robot0_MFJ1",
    "robot0_RFJ3",
    "robot0_RFJ2",
    "robot0_RFJ1",
    "robot0_LFJ4",
    "robot0_LFJ3",
    "robot0_LFJ2",
    "robot0_LFJ1",
    "robot0_THJ4",
    "robot0_THJ3",
    "robot0_THJ2",
    "robot0_THJ1",
    "robot0_THJ0",
]
"""Shadow Hand actuated joint names, in the Direct task's actuation order."""


def reorient_goal_command(**overrides) -> "mdp.ReorientEpisodeCommandCfg":
    """Build the object-pose goal command with the Direct-parity defaults.

    Keyword overrides replace the corresponding command fields; robot-specific
    variants declare the goal marker (``goal_pose_visualizer_cfg``) and their
    success threshold this way at the declaration site.
    """
    return mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=IN_HAND_POS_OFFSET,
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        debug_vis=True,
    ).replace(**overrides)


def reorient_reward_term(**param_overrides) -> RewTerm:
    """Build the :class:`~isaaclab_tasks.core.reorient.mdp.ReorientReward` term.

    The parameter defaults are the Direct-parity values of the Shadow Hand
    state task; keyword overrides are merged on top, so variants state only
    their deltas at the declaration site.
    """
    params = {
        "command_name": "object_pose",
        "distance_scale": -10.0,
        "rotation_scale": 1.0,
        "rotation_epsilon": 0.1,
        "action_penalty_scale": -0.0002,
        "success_tolerance": 0.1,
        "success_bonus": 250.0,
        "fall_distance": 0.24,
        "fall_penalty": 0.0,
        "averaging_factor": 0.1,
        "success_count_threshold": 1,
        "object_cfg": SceneEntityCfg("object"),
    }
    params.update(param_overrides)
    return RewTerm(func=mdp.ReorientReward, weight=1.0, params=params)


def reorient_joint_action(joint_names: list[str]) -> "mdp.EMAJointPositionToLimitsActionCfg":
    """Build the Direct-parity EMA joint-position action for the given joints."""
    return mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        alpha=1.0,
        rescale_to_limits=True,
    )


def reorient_reset_event(**param_overrides) -> EventTerm:
    """Build the Direct-parity state-reset event.

    The reset-noise defaults are shared by every reorientation task; keyword
    overrides are merged on top.
    """
    params = {
        "position_noise": 0.01,
        "joint_position_noise": 0.2,
        "joint_velocity_noise": 0.0,
        "action_name": "joint_pos",
    }
    params.update(param_overrides)
    return EventTerm(func=mdp.reset_reorient_state, mode="reset", params=params)


@configclass
class ReorientTerminationsCfg:
    """Termination conditions matching the Direct task."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": 0.24,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
