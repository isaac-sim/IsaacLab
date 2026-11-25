# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording action configuration for AllegroHand."""

from isaaclab_neural.record.recording_cfg import NeRDExplorationActionCfg, NeRDRecorderCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset

from ..allegro_hand_manager_env_cfg import AllegroHandManagerEnvCfg


@configclass
class ObservationsCfg:
    """Recording collects dynamics through recorder terms and needs no observation groups."""


@configclass
class AllegroHandNeRDExplorationCfg(NeRDExplorationActionCfg):
    """State exploration and joint excitation for recording."""

    class_type: str = "{DIR}.nerd_recording:AllegroHandNeRDExploration"
    episode_frames: tuple[int, int] = (1, 1)
    limit_margin: float = 0.15
    speeds: tuple[float, ...] = (30.0, 3.0, 0.3)
    target_reaches: tuple[float, ...] = (2.0, 0.3, 0.03)
    finger_joint_count: int = 4


_TASK = AllegroHandManagerEnvCfg()


@configclass
class AllegroHandRecordingActionsCfg:
    """Exploration joint targets."""

    joint_targets = AllegroHandNeRDExplorationCfg()


@configclass
class AllegroHandNeRDRecordingEnvCfg(ManagerBasedRLEnvCfg):
    """Reference scene, exploration terms and standard recorder terms."""

    recorders = NeRDRecorderCfg()
    scene = _TASK.scene
    sim = replace(
        _TASK.sim,
        dt=_TASK.sim.dt / _TASK.sim.physics.newton_mjwarp.num_substeps,
        render_interval=1,
        physics=preset(
            default=replace(
                _TASK.sim.physics.newton_mjwarp,
                num_substeps=1,
                solver_cfg=replace(_TASK.sim.physics.newton_mjwarp.solver_cfg, update_data_interval=1, njmax=256),
            ),
            newton_mjwarp=replace(
                _TASK.sim.physics.newton_mjwarp,
                num_substeps=1,
                solver_cfg=replace(_TASK.sim.physics.newton_mjwarp.solver_cfg, update_data_interval=1, njmax=256),
            ),
        ),
    )
    decimation = 1
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
    actions = AllegroHandRecordingActionsCfg()
    events = {
        "sample_robot": EventTermCfg(func="isaaclab_neural.record.recording:reset_exploration_state", mode="reset")
    }
