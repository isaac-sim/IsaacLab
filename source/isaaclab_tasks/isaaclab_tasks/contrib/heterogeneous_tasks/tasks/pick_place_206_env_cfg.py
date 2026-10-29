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
class PickPlace206EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the can of corn and put it in the metal bin, on a conference table."""

    instruction = "pick up the can of corn and put it in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object("steam", (2.30517, 0.02027, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True)
        self.scene.bin_b04 = library_object("bin_b04", (0.43107, -0.03559, 0.109), (0.0, 0.0, 0.662159, 0.749363))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.30801, -0.24693, 0.109), (0.0, 0.0, 0.0064, 0.99998)
        )
        self.scene.corn_can = library_object("corn_can", (0.45394, 0.26189, 0.13802), (0.0, 0.0, -0.999883, 0.015284))
        self.scene.milk_carton = library_object(
            "milk_carton", (0.3466, 0.2386, 0.20418), (0.0, 0.0, 0.753928, 0.656957)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.53244, 0.20266, 0.18295), (0.0, 0.0, 0.965164, 0.261645)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("corn_can")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
