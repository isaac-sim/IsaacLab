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
class PickPlace197EnvCfg(VLABenchmarkEnvCfg):
    """Put the sugar box on the book, on a hexagonal table."""

    instruction = "put the sugar box on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.75839, -0.06587, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_01 = library_object("av_book_01", (0.43159, -0.02415, 0.071), (0.0, 0.0, -0.99975, 0.022371))
        self.scene.sugar_box = library_object("sugar_box", (0.45428, -0.24831, 0.15901), (0.0, 0.0, 0.667326, 0.744766))
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.59248, -0.02448, 0.071), (0.0, 0.0, 0.990866, 0.13485)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.4327, 0.2476, 0.07108), (0.0, 0.0, -0.528312, 0.849051)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.3672, -0.28332, 0.10239), (0.0, 0.0, 0.13347, 0.991053)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.36481, 0.18292, 0.07106), (0.0, 0.0, -0.998134, 0.061059)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("sugar_box")],
                "reference": SceneEntityCfg("av_book_01"),
                "require_gripper_detached": True,
            },
        )
