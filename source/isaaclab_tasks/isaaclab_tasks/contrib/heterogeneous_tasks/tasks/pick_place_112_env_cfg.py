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
class PickPlace112EnvCfg(VLABenchmarkEnvCfg):
    """Move the yellow block into the salt bowl, on a table."""

    instruction = "move the yellow block into the salt bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.57749, -0.07961, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.35752, 0.07634, 0.00906), (0.0, 0.0, -0.080384, 0.996764)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.4665, 0.19284, 0.00899), (0.0, 0.0, 0.997589, 0.069397))
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.48122, 0.26553, 0.009), (0.0, 0.0, -0.365764, 0.930708)
        )
        self.scene.mug = library_object("mug", (0.54454, 0.00587, 0.00918), (0.0, 0.0, 0.318331, 0.947979))
        self.scene.spoon = library_object("spoon", (0.50304, -0.13661, 0.01778), (0.0, 0.0, 0.999272, 0.038163))
        self.scene.yellow_block = library_object(
            "yellow_block", (0.41017, -0.23808, 0.0325), (0.0, 0.0, -0.95821, 0.286067)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.33025, -0.21191, 0.09213), (0.0, 0.0, 0.999805, 0.019741)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("yellow_block")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
