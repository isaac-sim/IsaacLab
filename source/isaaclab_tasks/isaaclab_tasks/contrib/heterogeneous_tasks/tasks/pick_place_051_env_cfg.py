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
class PickPlace051EnvCfg(VLABenchmarkEnvCfg):
    """Put the butter on the book, on a table."""

    instruction = "put the butter on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.55714, -0.06288, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_21 = library_object("av_book_21", (0.43211, 0.08839, 0.009), (0.0, 0.0, -0.101104, 0.994876))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.butter = library_object("butter", (0.53885, -0.14717, 0.03541), (0.0, 0.0, 0.999994, 0.003401))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.blue_block = library_object(
            "blue_block", (0.41317, -0.26539, 0.0325), (0.0, 0.0, -0.536293, 0.844032)
        )
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.31832, -0.16426, 0.00901), (0.0, 0.0, -0.519475, 0.854486)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("butter")],
                "reference": SceneEntityCfg("av_book_21"),
                "require_gripper_detached": True,
            },
        )
