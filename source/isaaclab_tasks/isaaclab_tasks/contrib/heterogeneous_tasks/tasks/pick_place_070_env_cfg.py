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
class PickPlace070EnvCfg(VLABenchmarkEnvCfg):
    """Place both the granola bar box and the tomato sauce can into the black bowl, on a desk."""

    instruction = "place both the granola bar box and the tomato sauce can into the black bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.48613, 0.52518, -0.69646), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.43179, -0.08324, 0.08994), (0.0, 0.0, 0.620985, 0.783822)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.granola_bars = library_object(
            "granola_bars", (0.37507, 0.201, 0.152), (0.0, 0.0, -0.995238, 0.097473)
        )
        self.scene.av_apple = library_object("av_apple", (0.54312, 0.13169, 0.09096), (0.0, 0.0, 0.0396, 0.999216))
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.46564, 0.22522, 0.13142), (0.0, 0.0, 0.230018, 0.973186)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("granola_bars"), SceneEntityCfg("tomato_sauce_can")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "require_gripper_detached": True,
            },
        )
