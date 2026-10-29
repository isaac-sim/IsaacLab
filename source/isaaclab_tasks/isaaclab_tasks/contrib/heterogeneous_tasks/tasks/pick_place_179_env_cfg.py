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
class PickPlace179EnvCfg(VLABenchmarkEnvCfg):
    """Take the strawberry out of the red bowl, on an open shelf."""

    instruction = "take the strawberry out of the red bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.4156, -0.07807, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.37922, 0.06242, -0.00352), (0.0, 0.0, 0.934933, 0.354825))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.37922, 0.06242, -0.01719), (0.0, 0.0, -0.043481, 0.999054)
        )
        self.scene.mug = library_object("mug", (0.51623, -0.06317, -0.03082), (0.0, 0.0, 0.077316, 0.997007))
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.37437, -0.18136, -0.00497), (0.0, 0.0, -0.148266, 0.988947)
        )
        self.scene.bagel_00 = library_object("bagel_00", (0.48682, 0.19809, -0.01542), (0.0, 0.0, -0.957643, 0.287958))
        self.scene.mustard = library_object("mustard", (0.35893, 0.2557, 0.0647), (0.0, 0.0, 0.996255, 0.086458))
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("av_stawberry")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
