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
class PickPlace024EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the crab pen holder and put it in the black bowl, on a dining table."""

    instruction = "pick up the crab pen holder and put it in the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.55623, 0.55802, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.46558, 0.16135, 0.07096), (0.0, 0.0, 0.732819, 0.680423)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.56016, -0.05777, 0.07097), (0.0, 0.0, -0.807447, 0.58994)
        )
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.31873, 0.03434, 0.12243), (0.0, 0.0, 0.060928, 0.998142)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.3169, -0.23656, 0.08776), (0.0, 0.0, 0.359867, 0.933004))
        self.scene.spoon_1 = library_object("spoon_1", (0.39627, -0.03874, 0.08166), (0.0, 0.0, 0.343886, 0.939011))
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.29417, 0.14731, 0.071), (0.0, 0.0, -0.979981, 0.199089)
        )
        self.scene.pitcher_ycb = library_object(
            "pitcher_ycb", (0.4646, -0.20654, 0.19232), (0.0, 0.0, 0.169237, 0.985575)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("crabbypenholder")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
