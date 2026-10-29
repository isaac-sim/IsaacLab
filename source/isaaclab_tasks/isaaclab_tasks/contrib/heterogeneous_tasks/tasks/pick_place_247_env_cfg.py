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
class PickPlace247EnvCfg(VLABenchmarkEnvCfg):
    """Place the yellow block into the metal bin, on a storage cabinet."""

    instruction = "place the yellow block into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.45588, -0.0753, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.45876, 0.04938, -0.031), (0.0, 0.0, -0.740629, 0.671914))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.28677, -0.01083, 0.05701), (0.0, 0.0, 0.038131, 0.999273))
        self.scene.oatmeal_raisin_cookies = library_object(
            "oatmeal_raisin_cookies", (0.29654, 0.14275, 0.05262), (0.0, 0.0, 0.193703, 0.98106)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.31871, -0.25482, -0.01471), (0.0, 0.0, 0.994459, 0.105121)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.yellow_block = library_object(
            "yellow_block", (0.33772, 0.26201, -0.0075), (0.0, 0.0, -0.672512, 0.740086)
        )
        self.scene.brick = library_object("brick", (0.4593, -0.21101, -0.00543), (0.0, 0.0, -0.896594, 0.442853))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("yellow_block")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
