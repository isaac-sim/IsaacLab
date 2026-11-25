# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording action configuration for AnymalC."""

from isaaclab_neural.record.recording_cfg import NeRDExplorationActionCfg, NeRDRecorderCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset, resolve_presets

from ..flat_env_cfg import ANYMAL_C_NERD_TARGET_REACH, AnymalCFlatEnvCfg


@configclass
class ObservationsCfg:
    """Recording collects dynamics through recorder terms and needs no observation groups."""


@configclass
class AnymalCNeRDExplorationCfg(NeRDExplorationActionCfg):
    """State exploration and joint excitation for recording."""

    class_type: str = "{DIR}.nerd_recording:AnymalCNeRDExploration"
    episode_frames: tuple[int, int] = (100, 100)
    hold_frames: tuple[int, int] = (1, 10)
    drop: float = 0.2
    tilt: float = 0.5
    base_speed: tuple[float, float] = (1.0, 2.0)
    joint_reach: float = 0.5
    joint_speed: float = 2.0


_DECLARED = AnymalCFlatEnvCfg()
"""Task declaration, whose numerical physics alternatives a recording offers."""

_TASK = resolve_presets(AnymalCFlatEnvCfg(), selected=("newton_mjwarp",))
"""Task on its reference solver. The timing, actuators and events of its neural preset are no recording choices."""


@configclass
class AnymalCRecordingActionsCfg:
    """Exploration joint targets."""

    joint_targets = AnymalCNeRDExplorationCfg(
        clip={
            joints: (angle - ANYMAL_C_NERD_TARGET_REACH, angle + ANYMAL_C_NERD_TARGET_REACH)
            for joints, angle in _TASK.scene.robot.init_state.joint_pos.items()
        }
    )


@configclass
class AnymalCNeRDRecordingEnvCfg(ManagerBasedRLEnvCfg):
    """Reference scene, exploration terms and standard recorder terms."""

    recorders = NeRDRecorderCfg()
    scene = _TASK.scene
    sim = replace(
        _TASK.sim,
        physics=preset(
            default=_DECLARED.sim.physics.newton_mjwarp,
            newton_mjwarp=_DECLARED.sim.physics.newton_mjwarp,
            newton_steel=_DECLARED.sim.physics.newton_steel,
        ),
    )
    decimation = _TASK.decimation
    episode_length_s = _TASK.episode_length_s
    observations: ObservationsCfg = ObservationsCfg()
    rewards = {}
    terminations = {
        "invalid_state": TerminationTermCfg(func="isaaclab_neural.record.recording:invalid_exploration_state"),
        "exploration_timeout": TerminationTermCfg(
            func="isaaclab_neural.record.recording:exploration_timeout", time_out=True
        ),
    }
    commands = {}
    curriculum = {}
    actions = AnymalCRecordingActionsCfg()
    events = {
        "physics_material": _TASK.events.physics_material,
        "sample_robot": EventTermCfg(func="isaaclab_neural.record.recording:reset_exploration_state", mode="reset"),
    }
