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
class PickPlace156EnvCfg(VLABenchmarkEnvCfg):
    """Move the cookie box into the pan, on a storage unit."""

    instruction = "move the cookie box into the pan"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.62451, 0.14376, -0.6376), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.44507, -0.00649, 0.15626), (0.0, 0.0, -0.996384, 0.084959))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.oatmeal_raisin_cookies = library_object(
            "oatmeal_raisin_cookies", (0.29594, -0.2266, 0.23962), (0.0, 0.0, -0.982152, 0.188089)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.wood_block = library_object(
            "wood_block", (0.41504, -0.21151, 0.25894), (0.0, 0.0, 0.753835, 0.657063)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("oatmeal_raisin_cookies")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
