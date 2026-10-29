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
class PickPlace122EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the chess queen and put it in the metal bin, on a storage cabinet."""

    instruction = "pick up the chess queen and put it in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.49479, -0.07245, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.41333, -0.0891, -0.031), (0.0, 0.0, 0.772448, 0.635078))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.51195, 0.12339, -0.031), (0.0, 0.0, 0.538087, 0.842889)
        )
        self.scene.soup_can = library_object("soup_can", (0.3763, 0.15069, -0.03094), (0.0, 0.0, 0.96491, 0.262582))
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.38982, 0.251, 0.0433), (0.0, 0.0, -0.938555, 0.34513)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.57213, 0.02581, -0.00221), (0.0, 0.0, 0.61681, 0.787112)
        )
        self.scene.lime01 = library_object("lime01", (0.52013, 0.21347, -0.03078), (0.0, 0.0, 0.996864, 0.07914))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
