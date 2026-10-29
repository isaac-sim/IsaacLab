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
class PickPlace229EnvCfg(VLABenchmarkEnvCfg):
    """Place a piece of fruit into the white bowl, on a file cabinet."""

    instruction = "place a piece of fruit into the white bowl"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.43547, 0.09164, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.44429, -0.06319, -0.0312), (0.0, 0.0, 0.463414, 0.886142)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.38963, 0.14309, -0.03101), (0.0, 0.0, 0.888763, 0.458366))
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.44832, 0.23254, -0.03119), (0.0, 0.0, 0.089834, 0.995957)
        )
        self.scene.pumpkinlarge = library_object(
            "pumpkinlarge", (0.31594, 0.08085, -0.03124), (0.0, 0.0, 0.324649, 0.945834)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.avocado01 = library_object(
            "avocado01", (0.35628, -0.20478, -0.03117), (0.0, 0.0, -0.883553, 0.468332)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_stawberry"), SceneEntityCfg("av_kiwi01"), SceneEntityCfg("avocado01")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
