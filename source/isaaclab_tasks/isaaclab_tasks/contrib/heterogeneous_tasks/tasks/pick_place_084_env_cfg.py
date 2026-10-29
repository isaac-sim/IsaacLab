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
class PickPlace084EnvCfg(VLABenchmarkEnvCfg):
    """Move the can of corn into the pan, on a dining table."""

    instruction = "move the can of corn into the pan"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.58615, 0.45857, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_pan_b = library_object("av_pan_b", (0.40063, 0.09358, 0.07688), (0.0, 0.0, -0.014089, 0.999901))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.corn_can = library_object("corn_can", (0.33067, -0.2616, 0.10602), (0.0, 0.0, 0.998445, 0.055752))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("corn_can")],
                "container": SceneEntityCfg("av_pan_b"),
                "require_gripper_detached": True,
            },
        )
