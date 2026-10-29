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
class PickPlace158EnvCfg(VLABenchmarkEnvCfg):
    """Move the orange juice carton into the pan, on a table."""

    instruction = "move the orange juice carton into the pan"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.71526, -0.00197, -0.347), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_pan_b = library_object("av_pan_b", (0.40713, -0.12448, 0.00888), (0.0, 0.0, -0.999998, 0.001819))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.3286, 0.26095, 0.009), (0.0, 0.0, 0.178365, 0.983964)
        )
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.43454, 0.24251, 0.10524), (0.0, 0.0, 0.189675, 0.981847)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.yellow_block = library_object(
            "yellow_block", (0.58991, 0.01649, 0.0325), (0.0, 0.0, -0.999982, 0.006023)
        )
        self.scene.red_block = library_object("red_block", (0.54048, 0.22774, 0.0325), (0.0, 0.0, -0.999476, 0.032367))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_juice_carton")],
                "container": SceneEntityCfg("av_pan_b"),
                "require_gripper_detached": True,
            },
        )
