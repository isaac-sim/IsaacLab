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
class PickPlace118EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the can of corn and put it in the red bowl, on a table."""

    instruction = "pick up the can of corn and put it in the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.87585, 0.0617, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.50573, 0.01983, -0.03052), (0.0, 0.0, 0.333868, 0.94262))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.35696, 0.01002, -0.05802), (0.0, 0.0, 0.192699, 0.981258)
        )
        self.scene.glasses = library_object("glasses", (0.45065, -0.16807, -0.02753), (0.0, 0.0, 0.724602, 0.689167))
        self.scene.corn_can = library_object("corn_can", (0.29812, 0.13606, -0.02898), (0.0, 0.0, 0.50825, 0.861209))
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.50902, 0.22513, -0.04134), (0.0, 0.0, 0.959702, 0.281021)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.30008, 0.24884, -0.058), (0.0, 0.0, 0.115489, 0.993309)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("corn_can")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
