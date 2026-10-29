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
class PickPlace213EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the red block and put it in the metal bin, on a table."""

    instruction = "pick up the red block and put it in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.48618, -0.14997, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.38378, -0.10354, 0.009), (0.0, 0.0, 0.7386, 0.674144))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_12 = library_object(
            "spatula_12", (0.52887, 0.10145, 0.01267), (0.0, 0.0, 0.812213, 0.583361)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.mug_ycb = library_object("mug_ycb", (0.3435, 0.22596, 0.04969), (0.0, 0.0, 0.993658, 0.112447))
        self.scene.red_block = library_object("red_block", (0.38493, 0.12116, 0.0325), (0.0, 0.0, 0.701915, 0.71226))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("red_block")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
