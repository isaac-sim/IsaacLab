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
class PickPlace152EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the chess queen and put it in the metal bin, on an open shelf."""

    instruction = "pick up the chess queen and put it in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.41709, -0.03862, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.42853, -0.02403, -0.031), (0.0, 0.0, 0.481274, 0.87657))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.31396, 0.1359, -0.031), (0.0, 0.0, -0.998952, 0.04578)
        )
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.3027, -0.25368, -0.031), (0.0, 0.0, -0.682976, 0.730441)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.48921, -0.2504, -0.00221), (0.0, 0.0, -0.630846, 0.775908)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
