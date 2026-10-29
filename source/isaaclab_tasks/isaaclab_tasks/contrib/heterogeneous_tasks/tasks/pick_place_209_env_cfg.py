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
class PickPlace209EnvCfg(VLABenchmarkEnvCfg):
    """Take the strawberry out of the salt bowl, on a storage unit."""

    instruction = "take the strawberry out of the salt bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.62319, 0.07574, -0.6376), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.49573, -0.01769, 0.15606), (0.0, 0.0, -0.781901, 0.623403)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_10 = library_object(
            "spatula_10", (0.42169, -0.16687, 0.16273), (0.0, 0.0, 0.98927, 0.146099)
        )
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.49574, -0.01769, 0.16981), (0.0, 0.0, 0.826819, 0.562468)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.32443, 0.04479, 0.2303), (0.0, 0.0, -0.987999, 0.154463)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("av_stawberry")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
