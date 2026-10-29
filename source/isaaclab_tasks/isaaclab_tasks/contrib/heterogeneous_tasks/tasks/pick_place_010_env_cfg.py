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
class PickPlace010EnvCfg(VLABenchmarkEnvCfg):
    """Put the eraser on the book, on a table."""

    instruction = "put the eraser on the book"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.91518, -0.01244, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_66 = library_object("av_book_66", (0.41797, -0.13788, -0.058), (0.0, 0.0, 0.87137, 0.490626))
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.43496, 0.20795, -0.05803), (0.0, 0.0, -0.984093, 0.177655)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.53092, 0.0965, -0.058), (0.0, 0.0, -0.306477, 0.951878)
        )
        self.scene.spoon_2 = library_object("spoon_2", (0.52741, 0.23147, -0.05261), (0.0, 0.0, -0.719327, 0.694671))
        self.scene.av_eraser = library_object(
            "av_eraser", (0.40902, 0.15518, -0.05367), (0.0, 0.0, -0.236965, 0.971518)
        )
        self.scene.lime01 = library_object("lime01", (0.33294, 0.16537, -0.05778), (0.0, 0.0, 0.723683, 0.690133))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_eraser")],
                "reference": SceneEntityCfg("av_book_66"),
                "require_gripper_detached": True,
            },
        )
