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
class PickPlace043EnvCfg(VLABenchmarkEnvCfg):
    """Move the chess queen into the red bowl, on an end table."""

    instruction = "move the chess queen into the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.41982, 0.01126, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.36644, 0.0977, -0.02852), (0.0, 0.0, 0.711457, 0.70273))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_01 = library_object(
            "spatula_01", (0.54053, -0.00898, -0.04925), (0.0, 0.0, 0.73228, 0.681004)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.41318, 0.2208, -0.056), (0.0, 0.0, 0.111914, 0.993718)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.storage_box = library_object(
            "storage_box", (0.52004, 0.21129, -0.05613), (0.0, 0.0, 0.135865, 0.990727)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.30443, -0.04878, -0.056), (0.0, 0.0, -0.994142, 0.108078)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.41881, -0.13161, 0.01407), (0.0, 0.0, 0.980232, 0.197854)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
