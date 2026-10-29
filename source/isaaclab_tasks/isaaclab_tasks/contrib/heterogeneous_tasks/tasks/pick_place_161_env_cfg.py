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
class PickPlace161EnvCfg(VLABenchmarkEnvCfg):
    """Put the crab pen holder in the decorative bowl, on a table."""

    instruction = "put the crab pen holder in the decorative bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.56513, -0.03312, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.41456, 0.05956, 0.00904), (0.0, 0.0, 0.992954, 0.1185)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.46556, -0.24104, 0.00897), (0.0, 0.0, 0.984241, 0.176832)
        )
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.40284, 0.25599, 0.009), (0.0, 0.0, -0.409538, 0.912293)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_apple = library_object("av_apple", (0.4002, -0.17193, 0.00996), (0.0, 0.0, -0.459313, 0.888275))
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.32303, -0.23016, 0.05042), (0.0, 0.0, -0.941257, 0.337691)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("crabbypenholder")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
