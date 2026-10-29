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
class PickPlace100EnvCfg(VLABenchmarkEnvCfg):
    """Remove the wooden candle holder from the metal bin, on a conference table."""

    instruction = "remove the wooden candle holder from the metal bin"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.gilbert = library_object("gilbert", (1.64664, 0.08091, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True)
        self.scene.bin_a05 = library_object("bin_a05", (0.42849, -0.12845, 0.119), (0.0, 0.0, -0.669449, 0.742858))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.29641, 0.24566, 0.119), (0.0, 0.0, -0.752171, 0.658968)
        )
        self.scene.mug = library_object("mug", (0.42354, 0.22613, 0.11918), (0.0, 0.0, -0.041432, 0.999141))
        self.scene.spoon_big = library_object("spoon_big", (0.51813, 0.10186, 0.11924), (0.0, 0.0, 0.980438, 0.196826))
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.42849, -0.12845, 0.13301), (0.0, 0.0, 0.189047, 0.981968)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("av_woodenchmedium")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
