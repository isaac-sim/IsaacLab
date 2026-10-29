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
class PickPlace094EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess bishop in the metal bin, on an end table."""

    instruction = "put the chess bishop in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.4672, 0.02254, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.42875, 0.01483, -0.056), (0.0, 0.0, 0.792633, 0.609699))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.30141, -0.06864, -0.05603), (0.0, 0.0, -0.865014, 0.501748)
        )
        self.scene.butter = library_object("butter", (0.32843, 0.27845, -0.02959), (0.0, 0.0, -0.724437, 0.689341))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.43525, 0.23993, -0.056), (0.0, 0.0, -0.682463, 0.730921)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.31402, 0.22337, -0.03126), (0.0, 0.0, 0.796623, 0.604476)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_bishop")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
