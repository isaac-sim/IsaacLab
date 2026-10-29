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
class PickPlace023EnvCfg(VLABenchmarkEnvCfg):
    """Place the raisin box on top of the clay plate, on a table."""

    instruction = "place the raisin box on top of the clay plate"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.49563, 0.13217, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.clay_plates = library_object(
            "clay_plates", (0.40294, 0.12754, 0.00883), (0.0, 0.0, -0.999063, 0.043288)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.raisin_box = library_object(
            "raisin_box", (0.38731, -0.14867, 0.07059), (0.0, 0.0, 0.421142, 0.906995)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.29039, -0.14086, 0.009), (0.0, 0.0, -0.972369, 0.23345)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.51443, -0.17453, 0.03811), (0.0, 0.0, -0.287273, 0.957849)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("raisin_box")],
                "reference": SceneEntityCfg("clay_plates"),
                "require_gripper_detached": True,
            },
        )
