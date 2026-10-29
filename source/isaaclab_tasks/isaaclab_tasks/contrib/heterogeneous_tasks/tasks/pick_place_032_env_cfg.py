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
class PickPlace032EnvCfg(VLABenchmarkEnvCfg):
    """Place the white bottle into the salt bowl, on an open shelf."""

    instruction = "place the white bottle into the salt bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.44638, -0.02362, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.40279, 0.02017, -0.03094), (0.0, 0.0, 0.998369, 0.057087)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.41835, 0.23826, -0.031), (0.0, 0.0, 0.98238, 0.186894)
        )
        self.scene.mug = library_object("mug", (0.54737, 0.05852, -0.03082), (0.0, 0.0, 0.658822, 0.752299))
        self.scene.av_apple = library_object("av_apple", (0.32211, -0.18943, -0.03004), (0.0, 0.0, -0.04784, 0.998855))
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.54014, -0.19518, 0.04295), (0.0, 0.0, -0.990069, 0.140584)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
