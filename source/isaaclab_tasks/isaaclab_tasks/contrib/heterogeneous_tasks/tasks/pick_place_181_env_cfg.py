# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import HeterogeneousTaskManager, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..heterogeneous_tasks_env_cfg import VLABenchmarkEnvCfg
from ..objects import library_object


@configclass
class PickPlace181EnvCfg(VLABenchmarkEnvCfg):
    """Pick the white bottle out of the wooden bowl and put it down, on a conference table."""

    instruction = "pick the white bottle out of the wooden bowl and put it down"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.gilbert = library_object(
            "gilbert", (0.76538, 0.00768, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.40505, 0.0375, 0.11892), (0.0, 0.0, 0.705683, 0.708528)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.40692, 0.03853, 0.133), (0.0, 0.0, 0.758349, 0.651849)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.50571, -0.20421, 0.13529), (0.0, 0.0, 0.520544, 0.853835)
        )
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.41993, -0.17423, 0.119), (0.0, 0.0, -0.841249, 0.540648)
        )
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.37395, 0.2411, 0.11901), (0.0, 0.0, -0.320001, 0.947417)
        )
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.33704, -0.2431, 0.19316), (0.0, 0.0, 0.959541, 0.281568)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a02")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
