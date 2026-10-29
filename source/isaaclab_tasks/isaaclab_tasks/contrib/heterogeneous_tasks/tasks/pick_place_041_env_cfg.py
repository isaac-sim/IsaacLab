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
class PickPlace041EnvCfg(VLABenchmarkEnvCfg):
    """Put the spoon on the book, on a table."""

    instruction = "put the spoon on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.50502, -0.15541, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_01 = library_object(
            "av_book_01", (0.41223, -0.01119, 0.009), (0.0, 0.0, -0.073218, 0.997316)
        )
        self.scene.spatula_11 = library_object(
            "spatula_11", (0.39172, 0.22359, 0.01573), (0.0, 0.0, 0.722697, 0.691164)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon_1 = library_object("spoon_1", (0.57512, -0.12953, 0.01966), (0.0, 0.0, -0.76446, 0.644671))
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.41219, -0.25365, 0.009), (0.0, 0.0, 0.699508, 0.714625)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("spoon_1")],
                "reference": SceneEntityCfg("av_book_01"),
                "require_gripper_detached": True,
            },
        )
