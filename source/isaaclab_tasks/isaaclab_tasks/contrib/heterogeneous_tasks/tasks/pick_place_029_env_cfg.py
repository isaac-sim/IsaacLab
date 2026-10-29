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
class PickPlace029EnvCfg(VLABenchmarkEnvCfg):
    """Move the yellow block into the red bowl, on a console table."""

    instruction = "move the yellow block into the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.42634, -0.48953, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.423, -0.09271, 0.13648), (0.0, 0.0, -0.60546, 0.795876))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.fstylejug_a01 = library_object(
            "fstylejug_a01", (0.33335, -0.23985, 0.109), (0.0, 0.0, 0.689403, 0.724378)
        )
        self.scene.mug = library_object("mug", (0.37714, 0.21857, 0.10918), (0.0, 0.0, -0.155721, 0.987801))
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.34529, 0.02437, 0.10908), (0.0, 0.0, -0.998748, 0.050017)
        )
        self.scene.yellow_block = library_object(
            "yellow_block", (0.36244, 0.12374, 0.1325), (0.0, 0.0, 0.065589, 0.997847)
        )
        self.scene.potato_masher = library_object(
            "potato_masher", (0.48891, 0.10406, 0.10942), (0.0, 0.0, 0.169991, 0.985446)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("yellow_block")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
