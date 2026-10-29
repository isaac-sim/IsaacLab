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
class PickPlace047EnvCfg(VLABenchmarkEnvCfg):
    """Place the brick on the can of peaches, on a storage unit."""

    instruction = "place the brick on the can of peaches"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.29891, 0.24393, -0.6376), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object(
            "orange_02", (0.47493, -0.05949, 0.15606), (0.0, 0.0, -0.483402, 0.875399)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.32925, 3e-05, 0.18489), (0.0, 0.0, 0.121299, 0.992616)
        )
        self.scene.blue_block = library_object("blue_block", (0.37629, -0.09122, 0.1795), (0.0, 0.0, 0.57711, 0.816666))
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.52282, 0.0718, 0.15596), (0.0, 0.0, -0.393449, 0.919347)
        )
        self.scene.brick = library_object("brick", (0.33058, 0.16667, 0.18157), (0.0, 0.0, -0.326888, 0.945063))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("brick")],
                "reference": SceneEntityCfg("canned_peaches"),
                "require_gripper_detached": True,
            },
        )
