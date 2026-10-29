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
class PickPlace203EnvCfg(VLABenchmarkEnvCfg):
    """Move the white bottle into the metal bowl, on a storage cabinet."""

    instruction = "move the white bottle into the metal bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.44471, 0.11961, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.49441, -0.03098, -0.03075), (0.0, 0.0, 0.969535, 0.244953)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.43884, -0.1798, -0.031), (0.0, 0.0, -0.908775, 0.417286)
        )
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.30053, -0.1366, -0.031), (0.0, 0.0, -0.074502, 0.997221)
        )
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.40235, 0.16978, -0.00497), (0.0, 0.0, -0.989548, 0.144206)
        )
        self.scene.yellow_block = library_object(
            "yellow_block", (0.3456, 0.04403, -0.0075), (0.0, 0.0, 0.547002, 0.837131)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
