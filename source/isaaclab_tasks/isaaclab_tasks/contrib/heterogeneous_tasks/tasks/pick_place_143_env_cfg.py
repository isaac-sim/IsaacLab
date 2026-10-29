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
class PickPlace143EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the blue block and put it in the black bowl, on a console table."""

    instruction = "pick up the blue block and put it in the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.40532, -0.28405, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.43019, -0.04669, 0.19796), (0.0, 0.0, 0.168222, 0.985749)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cheez_it = library_object("cheez_it", (0.41462, 0.16598, 0.30474), (0.0, 0.0, -0.89188, 0.452272))
        self.scene.milkjug_a02 = library_object(
            "milkjug_a02", (0.47701, -0.23128, 0.198), (0.0, 0.0, 0.698133, 0.715968)
        )
        self.scene.blue_block = library_object(
            "blue_block", (0.29094, -0.24356, 0.2215), (0.0, 0.0, 0.630557, 0.776143)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.28783, -0.11219, 0.198), (0.0, 0.0, -0.991799, 0.127808)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("blue_block")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
