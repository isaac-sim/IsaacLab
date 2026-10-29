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
class PickPlace235EnvCfg(VLABenchmarkEnvCfg):
    """Move the orange juice carton into the metal bowl, on a hexagonal table."""

    instruction = "move the orange juice carton into the metal bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.74362, -0.04186, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.43357, 0.16841, 0.07125), (0.0, 0.0, -0.032992, 0.999456)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_13 = library_object(
            "spatula_13", (0.40472, -0.06189, 0.07467), (0.0, 0.0, -0.896482, 0.44308)
        )
        self.scene.butter = library_object("butter", (0.29279, -0.13705, 0.09741), (0.0, 0.0, -0.291454, 0.956585))
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.401, -0.20287, 0.16724), (0.0, 0.0, -0.215482, 0.976508)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.yellow_block = library_object(
            "yellow_block", (0.56927, 0.0212, 0.0945), (0.0, 0.0, 0.924958, 0.38007)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_juice_carton")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
