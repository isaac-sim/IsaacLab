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
class PickPlace114EnvCfg(VLABenchmarkEnvCfg):
    """Place both the kiwi and the orange juice carton into the metal bowl, on a storage unit."""

    instruction = "place both the kiwi and the orange juice carton into the metal bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.22922, -0.2144, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.45852, -0.10309, 0.15625), (0.0, 0.0, -0.749074, 0.662487)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.29494, -0.13545, 0.15599), (0.0, 0.0, 0.40249, 0.915424))
        self.scene.green_block = library_object(
            "green_block", (0.46632, -0.2351, 0.1795), (0.0, 0.0, -0.479213, 0.877699)
        )
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.51821, 0.05567, 0.25224), (0.0, 0.0, -0.736285, 0.676671)
        )
        self.scene.av_apple = library_object("av_apple", (0.50382, 0.18151, 0.15696), (0.0, 0.0, 0.566332, 0.824177))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.34224, 0.13632, 0.28132), (0.0, 0.0, -0.995765, 0.09193)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_kiwi01"), SceneEntityCfg("orange_juice_carton")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
