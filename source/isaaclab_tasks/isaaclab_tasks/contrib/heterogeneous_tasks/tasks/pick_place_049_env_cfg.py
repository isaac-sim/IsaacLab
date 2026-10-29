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
class PickPlace049EnvCfg(VLABenchmarkEnvCfg):
    """Put the spam can in the salt bowl, on a conference table."""

    instruction = "put the spam can in the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object(
            "steam", (0.72545, 0.55678, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.42206, 0.12062, 0.10906), (0.0, 0.0, -0.245395, 0.969423)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.27898, 0.22811, 0.10899), (0.0, 0.0, 0.731988, 0.681317))
        self.scene.spam_can = library_object("spam_can", (0.50532, -0.1864, 0.15082), (0.0, 0.0, -0.986003, 0.16673))
        self.scene.granola_bars = library_object(
            "granola_bars", (0.27914, 0.08804, 0.171), (0.0, 0.0, -0.056225, 0.998418)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.56869, 0.05648, 0.20418), (0.0, 0.0, 0.783495, 0.621399)
        )
        self.scene.naturalbostonroundbottle_a03 = library_object(
            "naturalbostonroundbottle_a03", (0.40383, -0.07814, 0.109), (0.0, 0.0, 0.994866, 0.101203)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.37703, -0.22595, 0.19213), (0.0, 0.0, 0.894742, 0.446584)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spam_can")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
