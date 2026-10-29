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
class PickPlace075EnvCfg(VLABenchmarkEnvCfg):
    """Move the green block into the salt bowl, on a storage unit."""

    instruction = "move the green block into the salt bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.62645, 0.17637, -0.6376), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.43453, 0.11341, 0.15606), (0.0, 0.0, -0.997954, 0.063935)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.32358, -0.16071, 0.15598), (0.0, 0.0, -0.908629, 0.417605)
        )
        self.scene.green_block = library_object(
            "green_block", (0.43989, -0.18695, 0.1795), (0.0, 0.0, -0.243762, 0.969835)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.measuring_cups = library_object(
            "measuring_cups", (0.45568, -0.046, 0.17495), (0.0, 0.0, -0.999887, 0.015029)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.31298, -0.03787, 0.23603), (0.0, 0.0, 0.530751, 0.847528)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("green_block")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
