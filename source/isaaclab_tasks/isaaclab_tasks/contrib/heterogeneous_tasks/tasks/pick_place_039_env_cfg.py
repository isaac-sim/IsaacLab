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
class PickPlace039EnvCfg(VLABenchmarkEnvCfg):
    """Put the orange can and the tomato sauce can in the metal bowl, on a conference table."""

    instruction = "put the orange can and the tomato sauce can in the metal bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.7233, 0.56238, -0.6971), (0.0, 0.0, 0.0, 1.0), kinematic=True)
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.37682, 0.13263, 0.12925), (0.0, 0.0, -0.485916, 0.874006)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.37666, -0.04829, 0.129), (0.0, 0.0, -0.00439, 0.99999)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.5378, 0.09058, 0.129), (0.0, 0.0, 0.662673, 0.748909)
        )
        self.scene.soup_can = library_object("soup_can", (0.46187, -0.21843, 0.12906), (0.0, 0.0, -0.191175, 0.981556))
        self.scene.av_apple = library_object("av_apple", (0.31996, -0.23288, 0.12996), (0.0, 0.0, -0.131641, 0.991298))
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.52347, -0.07304, 0.17042), (0.0, 0.0, -0.44462, 0.895719)
        )
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.51766, 0.20209, 0.25432), (0.0, 0.0, 0.440653, 0.897677)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("soup_can"), SceneEntityCfg("tomato_sauce_can")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
