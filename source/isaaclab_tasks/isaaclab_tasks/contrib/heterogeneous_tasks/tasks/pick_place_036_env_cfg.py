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
class PickPlace036EnvCfg(VLABenchmarkEnvCfg):
    """Pick the white bottle out of the black bowl and put it down, on a dining table."""

    instruction = "pick the white bottle out of the black bowl and put it down"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.60617, 0.02219, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.46981, 0.02805, 0.07096), (0.0, 0.0, 0.428648, 0.903472)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_05 = library_object(
            "spatula_05", (0.42168, -0.16402, 0.07471), (0.0, 0.0, 0.989055, 0.147548)
        )
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.46981, 0.02805, 0.085), (0.0, 0.0, -0.656198, 0.754588)
        )
        self.scene.av_hippoorganizer = library_object(
            "av_hippoorganizer", (0.37151, 0.17922, 0.071), (0.0, 0.0, 0.492001, 0.870594)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.52716, 0.20737, 0.16618), (0.0, 0.0, -0.819493, 0.573089)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a01")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
