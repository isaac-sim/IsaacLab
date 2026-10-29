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
class PickPlace115EnvCfg(VLABenchmarkEnvCfg):
    """Put a block in the metal bin, on a console table."""

    instruction = "put a block in the metal bin"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41549, 0.1191, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.34256, 0.22214, 0.069), (0.0, 0.0, -0.981187, 0.193061))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_10 = library_object(
            "spatula_10", (0.31375, -0.00651, 0.07573), (0.0, 0.0, -0.779123, 0.626871)
        )
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.54883, -0.21359, 0.0956), (0.0, 0.0, 0.985421, 0.170134)
        )
        self.scene.soup_can = library_object("soup_can", (0.52947, -0.0986, 0.06906), (0.0, 0.0, 0.989117, 0.14713))
        self.scene.blue_block = library_object("blue_block", (0.53371, 0.06742, 0.0925), (0.0, 0.0, -0.93555, 0.353194))
        self.scene.yellow_block = library_object(
            "yellow_block", (0.51364, 0.20489, 0.0925), (0.0, 0.0, -0.345714, 0.93834)
        )
        self.scene.brick = library_object("brick", (0.40044, -0.12902, 0.09457), (0.0, 0.0, -0.636448, 0.77132))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("blue_block"), SceneEntityCfg("brick"), SceneEntityCfg("yellow_block")],
                "container": SceneEntityCfg("bin_a02"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
