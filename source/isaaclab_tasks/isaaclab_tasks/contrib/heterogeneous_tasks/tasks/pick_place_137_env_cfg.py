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
class PickPlace137EnvCfg(VLABenchmarkEnvCfg):
    """Place the white bottle upright on top of the photo album, on a storage cabinet."""

    instruction = "place the white bottle upright on top of the photo album"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.42764, -0.06334, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.37171, 0.12555, -0.00352), (0.0, 0.0, -0.001011, 0.999999))
        self.scene.av_photoalbum_closed = library_object(
            "av_photoalbum_closed", (0.41701, -0.09691, -0.031), (0.0, 0.0, 0.583999, 0.811755)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.51019, 0.24452, -0.031), (0.0, 0.0, 0.01073, 0.999942)
        )
        self.scene.fork_big = library_object("fork_big", (0.26582, -0.11359, -0.03065), (0.0, 0.0, 0.713067, 0.701096))
        self.scene.blue_block = library_object(
            "blue_block", (0.32508, 0.25609, -0.0075), (0.0, 0.0, 0.470691, 0.882298)
        )
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.54646, -0.21251, -0.031), (0.0, 0.0, 0.975367, 0.220587)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                "reference": SceneEntityCfg("av_photoalbum_closed"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
