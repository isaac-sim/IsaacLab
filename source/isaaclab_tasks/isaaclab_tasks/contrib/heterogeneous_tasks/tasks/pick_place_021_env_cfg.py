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
class PickPlace021EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the pumpkin and put it in the salt bowl, on a file cabinet."""

    instruction = "pick up the pumpkin and put it in the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.48661, -0.06001, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.41963, -0.1915, -0.03094), (0.0, 0.0, -0.719968, 0.694008)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_09 = library_object(
            "spatula_09", (0.35502, 0.10313, -0.02427), (0.0, 0.0, 0.733855, 0.679307)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.50906, 0.15147, -0.03126), (0.0, 0.0, -0.981411, 0.191915)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.5648, 0.0235, -0.031), (0.0, 0.0, 0.908613, 0.417639)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.52168, -0.06688, 0.00039), (0.0, 0.0, 0.213532, 0.976936)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("pumpkinsmall")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
