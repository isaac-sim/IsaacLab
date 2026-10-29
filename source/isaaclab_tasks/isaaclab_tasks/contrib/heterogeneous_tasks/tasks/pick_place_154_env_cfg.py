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
class PickPlace154EnvCfg(VLABenchmarkEnvCfg):
    """Put the white bottle in the white bowl, on a table."""

    instruction = "put the white bottle in the white bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.50543, 0.15732, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.40127, -0.09698, 0.0088), (0.0, 0.0, 0.08179, 0.99665)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_06 = library_object("spatula_06", (0.42131, 0.10735, 0.01271), (0.0, 0.0, -0.0467, 0.998909))
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.32406, 0.22914, 0.009), (0.0, 0.0, 0.296532, 0.955023)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.57744, -0.07105, 0.009), (0.0, 0.0, 0.217819, 0.975989)
        )
        self.scene.brick = library_object("brick", (0.44568, 0.23002, 0.03457), (0.0, 0.0, 0.638195, 0.769875))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a03")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
