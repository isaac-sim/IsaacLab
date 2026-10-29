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
class PickPlace205EnvCfg(VLABenchmarkEnvCfg):
    """Pick the kiwi out of the metal bin and put it down, on a storage unit."""

    instruction = "pick the kiwi out of the metal bin and put it down"
    episode_length_s = 45.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.30996, 0.22021, -0.6376), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.33401, -0.0788, 0.156), (0.0, 0.0, 0.073271, 0.997312))
        self.scene.av_photoalbum_closed = library_object(
            "av_photoalbum_closed", (0.49503, 0.06343, 0.156), (0.0, 0.0, 0.866847, 0.498573)
        )
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.33401, -0.0788, 0.16999), (0.0, 0.0, -0.793014, 0.609204))
        self.scene.yogurt_cup = library_object("yogurt_cup", (0.32492, 0.14956, 0.18284), (0.0, 0.0, 0.99084, 0.135038))
        self.scene.tuna_can = library_object("tuna_can", (0.54544, -0.10995, 0.17276), (0.0, 0.0, -0.108578, 0.994088))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.30643, 0.04945, 0.16544), (0.0, 0.0, 0.370825, 0.928703)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
