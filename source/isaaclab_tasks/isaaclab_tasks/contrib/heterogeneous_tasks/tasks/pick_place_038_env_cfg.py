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
class PickPlace038EnvCfg(VLABenchmarkEnvCfg):
    """Move the ketchup bottle into the metal bin, on a table."""

    instruction = "move the ketchup bottle into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.50623, -0.03323, -0.69705), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.49828, -0.15607, 0.069), (0.0, 0.0, 0.660024, 0.751245))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_11 = library_object(
            "spatula_11", (0.39025, -0.08034, 0.07573), (0.0, 0.0, -0.214615, 0.976699)
        )
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.3959, 0.16606, 0.09584), (0.0, 0.0, -0.775041, 0.631911)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.46475, 0.24225, 0.1433), (0.0, 0.0, -0.598416, 0.801186)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.30036, 0.19435, 0.14295), (0.0, 0.0, 0.340636, 0.940195)
        )
        self.scene.potato_masher = library_object(
            "potato_masher", (0.52779, 0.07934, 0.06942), (0.0, 0.0, 0.986298, 0.164974)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("ketchup_bottle")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
