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
class PickPlace103EnvCfg(VLABenchmarkEnvCfg):
    """Put the white bottle and the can of peaches in the metal bin, on a conference table."""

    instruction = "put the white bottle and the can of peaches in the metal bin"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object("steam", (2.2874, 0.12701, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.bin_a05 = library_object("bin_a05", (0.43592, -0.04726, 0.109), (0.0, 0.0, 0.723402, 0.690427))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.44884, -0.25241, 0.109), (0.0, 0.0, -0.103497, 0.99463)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.32935, 0.20581, 0.13789), (0.0, 0.0, -0.674868, 0.737939)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.45732, 0.1624, 0.13374), (0.0, 0.0, 0.778857, 0.627202)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a03"), SceneEntityCfg("canned_peaches")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
