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
class PickPlace238EnvCfg(VLABenchmarkEnvCfg):
    """Pick the chocolate pudding mix out of the salt bowl and put it down, on a storage unit."""

    instruction = "pick the chocolate pudding mix out of the salt bowl and put it down"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.23829, -0.11929, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.50627, -0.03753, 0.15606), (0.0, 0.0, -0.524411, 0.851465)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_12 = library_object(
            "spatula_12", (0.35709, -0.03155, 0.15967), (0.0, 0.0, 0.758911, 0.651194)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinlarge = library_object(
            "pumpkinlarge", (0.49767, 0.12677, 0.15576), (0.0, 0.0, -0.935961, 0.352103)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.50627, -0.03752, 0.19474), (0.0, 0.0, -0.302792, 0.953057)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
