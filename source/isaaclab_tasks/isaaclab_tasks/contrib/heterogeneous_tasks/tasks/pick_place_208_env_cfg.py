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
class PickPlace208EnvCfg(VLABenchmarkEnvCfg):
    """Place the kiwi into the wooden bowl, on a conference table."""

    instruction = "place the kiwi into the wooden bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.gilbert = library_object(
            "gilbert", (0.77784, 0.10032, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.43859, 0.05503, 0.11892), (0.0, 0.0, 0.144946, 0.98944)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.42452, -0.21539, 0.11899), (0.0, 0.0, 0.07362, 0.997286))
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.33324, -0.1878, 0.11881), (0.0, 0.0, 0.549749, 0.83533)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
