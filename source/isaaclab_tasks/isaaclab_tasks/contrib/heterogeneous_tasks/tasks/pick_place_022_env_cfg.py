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
class PickPlace022EnvCfg(VLABenchmarkEnvCfg):
    """Place the eraser on top of the clay plate, on a conference table."""

    instruction = "place the eraser on top of the clay plate"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object(
            "steam", (0.75559, -0.95548, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.clay_plates = library_object(
            "clay_plates", (0.44751, 0.08788, 0.10883), (0.0, 0.0, 0.785448, 0.618927)
        )
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.30838, -0.19133, 0.10897), (0.0, 0.0, -0.416861, 0.90897)
        )
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.39111, -0.17986, 0.109), (0.0, 0.0, -0.311669, 0.950191)
        )
        self.scene.milkjug_a02 = library_object(
            "milkjug_a02", (0.49861, -0.17209, 0.109), (0.0, 0.0, 0.729794, 0.683667)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_eraser = library_object(
            "av_eraser", (0.36958, -0.28088, 0.11333), (0.0, 0.0, -0.994964, 0.100233)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_eraser")],
                "reference": SceneEntityCfg("clay_plates"),
                "require_gripper_detached": True,
            },
        )
