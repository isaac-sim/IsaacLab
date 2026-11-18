# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The stock single-camera PPO with gaze supervision and a fixed gaze noise; the actor and critic also see the gaze."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg import (
    CAMERA_ALGO_CFG,
    CNN_POLICY_CFG,
    KukaAllegroSingleCameraPPORunnerCfg,
)

from ..agents import gaze_actor, gaze_supervised


@configclass
class KukaAllegroFoviPPORunnerCfg(KukaAllegroSingleCameraPPORunnerCfg):
    experiment_name = "kuka_allegro_fovi"
    obs_groups = {
        "actor": ["policy", "proprio", "base_image", "gaze"],
        "critic": ["policy", "proprio", "perception", "gaze"],
    }
    actor = gaze_actor(CNN_POLICY_CFG)
    algorithm = gaze_supervised(CAMERA_ALGO_CFG)
