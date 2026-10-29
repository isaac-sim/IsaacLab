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
class PickPlace164EnvCfg(VLABenchmarkEnvCfg):
    """Take the marker out of the white bowl, on a console table."""

    instruction = "take the marker out of the white bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.43923, -0.34899, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.38329, 0.01646, 0.1088), (0.0, 0.0, -0.997277, 0.073745)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_18 = library_object(
            "spatula_18", (0.43335, -0.20931, 0.11268), (0.0, 0.0, 0.137231, 0.990539)
        )
        self.scene.spam_can = library_object("spam_can", (0.31245, 0.22972, 0.15082), (0.0, 0.0, -0.722717, 0.691144))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.43226, 0.22029, 0.109), (0.0, 0.0, -0.851298, 0.524683)
        )
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.38343, 0.01648, 0.13244), (0.0, 0.0, -0.330835, 0.943689)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.56324, 0.08439, 0.19213), (0.0, 0.0, 0.158781, 0.987314)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("dry_erase_marker")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
