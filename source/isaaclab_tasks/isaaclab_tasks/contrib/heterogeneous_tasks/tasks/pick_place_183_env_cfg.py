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
class PickPlace183EnvCfg(VLABenchmarkEnvCfg):
    """Put the butter on the book, on a storage unit."""

    instruction = "put the butter on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.22844, -0.19154, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_99 = library_object("av_book_99", (0.42785, 0.13376, 0.156), (0.0, 0.0, 0.987548, 0.157318))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.butter = library_object("butter", (0.47166, -0.18217, 0.18241), (0.0, 0.0, 0.48189, 0.876232))
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.33691, -0.10665, 0.156), (0.0, 0.0, -0.991886, 0.127132)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.29944, 0.20424, 0.156), (0.0, 0.0, -0.033683, 0.999433)
        )
        self.scene.av_eraser = library_object("av_eraser", (0.39259, -0.19287, 0.16033), (0.0, 0.0, 0.823914, 0.566715))
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.50148, -0.27141, 0.23603), (0.0, 0.0, -0.679686, 0.733503)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("butter")],
                "reference": SceneEntityCfg("av_book_99"),
                "require_gripper_detached": True,
            },
        )
