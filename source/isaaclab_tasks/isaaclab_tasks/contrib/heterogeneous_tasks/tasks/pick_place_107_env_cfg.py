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
class PickPlace107EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the plastic bottle and put it in the salt bowl, on a table."""

    instruction = "pick up the plastic bottle and put it in the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.62503, -0.00017, -0.347), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.37465, 0.06965, 0.00906), (0.0, 0.0, -0.053916, 0.998545)
        )
        self.scene.av_photoalbum_closed = library_object(
            "av_photoalbum_closed", (0.46426, -0.14334, 0.009), (0.0, 0.0, -0.949544, 0.313633)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.37427, 0.24433, 0.009), (0.0, 0.0, 0.958302, 0.285756)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon_big = library_object("spoon_big", (0.296, -0.132, 0.00924), (0.0, 0.0, -0.733466, 0.679726))
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.51781, 0.14257, 0.009), (0.0, 0.0, -0.977838, 0.209365)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
