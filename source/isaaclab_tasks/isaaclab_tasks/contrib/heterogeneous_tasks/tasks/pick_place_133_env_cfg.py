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
class PickPlace133EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the white bottle and put it in the metal bin, on a dining table."""

    instruction = "pick up the white bottle and put it in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.59735, -0.29278, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.46626, 0.16667, 0.071), (0.0, 0.0, -0.766414, 0.642347))
        self.scene.av_book_11 = library_object(
            "av_book_11", (0.43911, -0.13404, 0.071), (0.0, 0.0, -0.851137, 0.524944)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.53017, 0.00815, 0.071), (0.0, 0.0, -0.074767, 0.997201)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.31812, 0.11521, 0.08776), (0.0, 0.0, 0.472734, 0.881205))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.36002, 0.25523, 0.15103), (0.0, 0.0, 0.824594, 0.565725)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a03")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
