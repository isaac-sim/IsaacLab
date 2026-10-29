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
class PickPlace192EnvCfg(VLABenchmarkEnvCfg):
    """Place both the orange juice carton and the avocado into the red bowl, on a dining table."""

    instruction = "place both the orange juice carton and the avocado into the red bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.59527, 0.67429, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.45365, 0.12397, 0.10448), (0.0, 0.0, 0.66716, 0.744915))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.fstylejug_a02 = library_object(
            "fstylejug_a02", (0.50352, -0.06609, 0.077), (0.0, 0.0, 0.917333, 0.398122)
        )
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.33552, -0.04459, 0.17324), (0.0, 0.0, -0.665767, 0.74616)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.yellow_block = library_object(
            "yellow_block", (0.32577, 0.15766, 0.1005), (0.0, 0.0, -0.231222, 0.972901)
        )
        self.scene.avocado01 = library_object("avocado01", (0.4662, -0.25929, 0.07683), (0.0, 0.0, -0.808088, 0.589062))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_juice_carton"), SceneEntityCfg("avocado01")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
