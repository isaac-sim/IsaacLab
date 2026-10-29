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
class PickPlace215EnvCfg(VLABenchmarkEnvCfg):
    """Pick the can of pineapple out of the black bowl and put it down, on a table."""

    instruction = "pick the can of pineapple out of the black bowl and put it down"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.49635, -0.18411, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.39361, 0.06062, 0.00896), (0.0, 0.0, -0.480595, 0.876942)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.39361, 0.06062, 0.05181), (0.0, 0.0, 0.582234, 0.813021)
        )
        self.scene.spoon_big = library_object("spoon_big", (0.35137, 0.23356, 0.00924), (0.0, 0.0, 0.036913, 0.999318))
        self.scene.naturalbostonroundbottle_a03 = library_object(
            "naturalbostonroundbottle_a03", (0.51454, -0.11288, 0.009), (0.0, 0.0, 0.92636, 0.37664)
        )
        self.scene.mustard = library_object("mustard", (0.37912, -0.23121, 0.1047), (0.0, 0.0, -0.637432, 0.770507))
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("pineapple_slices_can")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
