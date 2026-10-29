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
class PickPlace035EnvCfg(VLABenchmarkEnvCfg):
    """Move the frozen waffles into the metal bin, on a hexagonal table."""

    instruction = "move the frozen waffles into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.78855, -0.09558, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.45751, -0.10011, 0.071), (0.0, 0.0, -0.822488, 0.568782))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.46308, 0.15109, 0.07098), (0.0, 0.0, -0.649644, 0.760238)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.32933, -0.26181, 0.1453), (0.0, 0.0, 0.689127, 0.724641)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.30866, -0.04399, 0.09574), (0.0, 0.0, 0.25941, 0.965767)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("frozen_waffles")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
