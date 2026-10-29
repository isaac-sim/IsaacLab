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
class PickPlace042EnvCfg(VLABenchmarkEnvCfg):
    """Put a block in the metal bowl, on an office desk."""

    instruction = "put a block in the metal bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.51427, -0.19626, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.4622, 0.09896, 0.00725), (0.0, 0.0, 0.28506, 0.95851)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_17 = library_object(
            "spatula_17", (0.36169, -0.09216, 0.01068), (0.0, 0.0, -0.605446, 0.795886)
        )
        self.scene.green_block = library_object(
            "green_block", (0.30387, -0.18282, 0.0305), (0.0, 0.0, -0.06623, 0.997804)
        )
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.35248, 0.21588, 0.007), (0.0, 0.0, -0.93402, 0.35722)
        )
        self.scene.blue_block = library_object("blue_block", (0.52018, -0.09974, 0.0305), (0.0, 0.0, 0.741176, 0.67131))
        self.scene.yellow_block = library_object(
            "yellow_block", (0.56683, -0.01712, 0.0305), (0.0, 0.0, 0.959696, 0.281042)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("yellow_block"),
                    SceneEntityCfg("green_block"),
                    SceneEntityCfg("blue_block"),
                ],
                "container": SceneEntityCfg("serving_bowl"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
