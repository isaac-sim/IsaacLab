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
class PickPlace186EnvCfg(VLABenchmarkEnvCfg):
    """Set the tomato soup can upright on the coaster, on a table."""

    instruction = "set the tomato soup can upright on the coaster"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.55779, 0.03555, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.40038, -0.10384, 0.009), (0.0, 0.0, -0.324421, 0.945913)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.40642, 0.24502, 0.06002), (0.0, 0.0, 0.933288, 0.359129)
        )
        self.scene.wood_block = library_object(
            "wood_block", (0.30954, 0.15442, 0.11194), (0.0, 0.0, -0.731918, 0.681393)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.5126, -0.23015, 0.009), (0.0, 0.0, -0.108774, 0.994067)
        )
        self.scene.birdhouse = library_object("birdhouse", (0.45882, 0.0763, 0.00905), (0.0, 0.0, 0.965926, 0.258817))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("tomato_soup_can")],
                "reference": SceneEntityCfg("av_coaster_hexagon"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
