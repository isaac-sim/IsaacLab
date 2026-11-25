# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording action configuration for Franka."""

from isaaclab_neural.record.recording_cfg import NeRDExplorationActionCfg, NeRDRecorderCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset

from .franka_reach_env_cfg import FrankaReachBaseEnvCfg


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


_TASK = FrankaReachBaseEnvCfg()


@configclass
class FrankaRecordingActionsCfg:
    """Exploration joint targets."""

    joint_targets = FrankaNeRDExplorationCfg()


@configclass
class FrankaNeRDRecordingEnvCfg(ManagerBasedRLEnvCfg):
    """Reference scene, exploration terms and standard recorder terms."""

    recorders = NeRDRecorderCfg()
    scene = replace(_TASK.scene, table=None)
    sim = replace(
        _TASK.sim,
        physics=preset(
            default=_TASK.sim.physics.newton_mjwarp,
            newton_mjwarp=_TASK.sim.physics.newton_mjwarp,
            newton_steel=_TASK.sim.physics.newton_steel,
        ),
    )
    decimation = _TASK.decimation
    episode_length_s = _TASK.episode_length_s
    observations = {}
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
