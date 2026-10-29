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
class PickPlace131EnvCfg(VLABenchmarkEnvCfg):
    """Place both the spoon and the cookie box into the black bowl, on a table."""

    instruction = "place both the spoon and the cookie box into the black bowl"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.64605, 0.02901, -0.347), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.43004, -0.02348, 0.00894), (0.0, 0.0, -0.047662, 0.998864)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.oatmeal_raisin_cookies = library_object(
            "oatmeal_raisin_cookies", (0.40404, 0.22107, 0.09262), (0.0, 0.0, -0.535856, 0.84431)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon_1 = library_object("spoon_1", (0.36259, -0.23529, 0.01966), (0.0, 0.0, 0.9817, 0.190436))
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.32854, 0.26285, 0.00901), (0.0, 0.0, 0.237583, 0.971367)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_1"), SceneEntityCfg("oatmeal_raisin_cookies")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "require_gripper_detached": True,
            },
        )
