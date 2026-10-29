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
class PickPlace066EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the spam can and put it in the red bowl, on a conference table."""

    instruction = "pick up the spam can and put it in the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.69751, -0.14768, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.bowl = library_object("bowl", (0.43468, -0.01013, 0.15648), (0.0, 0.0, 0.873654, 0.486547))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.spam_can = library_object("spam_can", (0.39519, 0.22249, 0.17082), (0.0, 0.0, 0.818424, 0.574615))
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.37853, -0.21045, 0.129), (0.0, 0.0, 0.115995, 0.99325)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.54265, 0.0885, 0.129), (0.0, 0.0, -0.656984, 0.753905)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.30013, 0.17836, 0.15779), (0.0, 0.0, 0.693433, 0.720521)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spam_can")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
