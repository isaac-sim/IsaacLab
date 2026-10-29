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
class PickPlace081EnvCfg(VLABenchmarkEnvCfg):
    """Place a block into the black bowl, on an end table."""

    instruction = "place a block into the black bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.45726, 0.0028, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.50588, 0.05343, -0.05604), (0.0, 0.0, -0.989058, 0.147525)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.41386, -0.21982, -0.0325), (0.0, 0.0, -0.121387, 0.992605)
        )
        self.scene.jello = library_object("jello", (0.43748, -0.13662, -0.04094), (0.0, 0.0, 0.643145, 0.765745))
        self.scene.blue_block = library_object(
            "blue_block", (0.33022, -0.19389, -0.0325), (0.0, 0.0, -0.935561, 0.353165)
        )
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.36901, 0.23057, -0.04166), (0.0, 0.0, 0.999234, 0.03913)
        )
        self.scene.red_block = library_object("red_block", (0.35361, 0.14214, -0.0325), (0.0, 0.0, 0.484398, 0.874848))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("green_block"), SceneEntityCfg("blue_block"), SceneEntityCfg("red_block")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
