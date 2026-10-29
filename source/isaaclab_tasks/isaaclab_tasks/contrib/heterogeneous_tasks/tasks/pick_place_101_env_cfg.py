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
class PickPlace101EnvCfg(VLABenchmarkEnvCfg):
    """Put the green block in the red bowl, on a conference table."""

    instruction = "put the green block in the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object(
            "steam", (0.77704, 1.6845, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.42169, 0.00092, 0.13648), (0.0, 0.0, -0.332865, 0.942974))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.51527, 0.20452, 0.1325), (0.0, 0.0, 0.923829, 0.382807)
        )
        self.scene.mug = library_object("mug", (0.32667, 0.22395, 0.10918), (0.0, 0.0, 0.938893, 0.344208))
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.33781, -0.20168, 0.109), (0.0, 0.0, -0.851321, 0.524645)
        )
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.44195, -0.22717, 0.109), (0.0, 0.0, 0.532985, 0.846125)
        )
        self.scene.potato_masher = library_object(
            "potato_masher", (0.53068, -0.13715, 0.10942), (0.0, 0.0, -0.582293, 0.812979)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("green_block")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
