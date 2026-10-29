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
class PickPlace166EnvCfg(VLABenchmarkEnvCfg):
    """Set the spoon on the book, on a dining table."""

    instruction = "set the spoon on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.69355, -0.41182, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_99 = library_object("av_book_99", (0.36283, -0.16156, 0.071), (0.0, 0.0, 0.002104, 0.999998))
        self.scene.spatula_18 = library_object(
            "spatula_18", (0.40111, 0.09367, 0.07468), (0.0, 0.0, 0.951209, 0.308546)
        )
        self.scene.fstylejug_a02 = library_object(
            "fstylejug_a02", (0.53986, -0.10117, 0.071), (0.0, 0.0, 0.997814, 0.06609)
        )
        self.scene.mug = library_object("mug", (0.41202, 0.21883, 0.07118), (0.0, 0.0, -0.787139, 0.616775))
        self.scene.spoon = library_object("spoon", (0.50604, 0.1955, 0.07978), (0.0, 0.0, 0.67633, 0.736599))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("spoon")],
                "reference": SceneEntityCfg("av_book_99"),
                "require_gripper_detached": True,
            },
        )
