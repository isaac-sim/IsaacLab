# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording action configuration for Franka."""

from isaaclab_neural.record.recording_cfg import NeRDExplorationActionCfg, NeRDRecorderCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset, resolve_presets

from ..franka_reach_env_cfg import FrankaReachEnvCfg


@configclass
class ObservationsCfg:
    """Recording collects dynamics through recorder terms and needs no observation groups."""


@configclass
class FrankaNeRDExplorationCfg(NeRDExplorationActionCfg):
    """State exploration and joint excitation for recording."""

    class_type: str = "{DIR}.nerd_recording:FrankaNeRDExploration"
    episode_frames: tuple[int, int] = (17, 64)
    arm_joint_count: int = 7
    state_speed: float = 1.0
    target_margin: float = 0.5
    hold_frames: tuple[int, int] = (1, 60)
    explore_reach: float = 1.0
    refine_reach: float = 0.3
    drift_timescale: float = 100.0


_DECLARED = FrankaReachEnvCfg()
"""Task declaration, whose numerical physics alternatives a recording offers."""

_TASK = resolve_presets(FrankaReachEnvCfg(), selected=("newton_mjwarp",))
"""Task on its reference solver. Its controller, holding and neural presets are not choices of a recording."""

_SCENE = replace(_TASK.scene, table=None)
# The physics variant of the robot asset follows the solver that is recorded.
_SCENE.robot.spawn.variants["Physics"] = _DECLARED.scene.robot.spawn.variants["Physics"]


@configclass
class FrankaRecordingActionsCfg:
    """Exploration joint targets."""

    joint_targets = FrankaNeRDExplorationCfg()


@configclass
class FrankaNeRDRecordingEnvCfg(ManagerBasedRLEnvCfg):
    """Reference scene, exploration terms and standard recorder terms."""

    recorders = NeRDRecorderCfg()
    scene = _SCENE
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
    actions = FrankaRecordingActionsCfg()
    events = {
        "sample_robot": EventTermCfg(func="isaaclab_neural.record.recording:reset_exploration_state", mode="reset")
    }
