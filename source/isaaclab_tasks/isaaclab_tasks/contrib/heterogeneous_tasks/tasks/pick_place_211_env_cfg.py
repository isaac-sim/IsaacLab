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
class PickPlace211EnvCfg(VLABenchmarkEnvCfg):
    """Put a bottle in the red bowl, on a table."""

    instruction = "put a bottle in the red bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.52647, 0.17149, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.46334, 0.08106, 0.03648), (0.0, 0.0, 0.71163, 0.702554))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.4456, 0.22878, 0.09701), (0.0, 0.0, -0.888674, 0.458539))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.52261, -0.08551, 0.009), (0.0, 0.0, 0.011301, 0.999936)
        )
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.30207, 0.11255, 0.0833), (0.0, 0.0, 0.807025, 0.590517)
        )
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.4099, -0.25314, 0.00901), (0.0, 0.0, 0.629991, 0.776602)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.37245, -0.09228, 0.08903), (0.0, 0.0, 0.91061, 0.413266)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [
                    SceneEntityCfg("milkjug_a01"),
                    SceneEntityCfg("ketchup_bottle"),
                    SceneEntityCfg("mustard_bottle"),
                ],
                "container": SceneEntityCfg("bowl"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
