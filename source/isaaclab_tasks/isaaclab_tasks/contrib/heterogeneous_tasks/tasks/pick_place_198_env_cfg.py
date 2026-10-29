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
class PickPlace198EnvCfg(VLABenchmarkEnvCfg):
    """Move the green block into the white bowl, on a dining table."""

    instruction = "move the green block into the white bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.55481, -0.34466, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.41523, -0.11869, 0.0708), (0.0, 0.0, 0.818437, 0.574596)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.53487, 0.08586, 0.07098), (0.0, 0.0, -0.998648, 0.051985)
        )
        self.scene.green_block = library_object(
            "green_block", (0.42311, 0.1449, 0.0945), (0.0, 0.0, -0.909779, 0.415093)
        )
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.30152, 0.0834, 0.09981), (0.0, 0.0, -0.896914, 0.442205)
        )
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.53499, 0.23462, 0.1453), (0.0, 0.0, 0.879981, 0.475009)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.37916, 0.04878, 0.071), (0.0, 0.0, 0.593341, 0.804951)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.36796, 0.20839, 0.15103), (0.0, 0.0, -0.907736, 0.419542)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("green_block")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
