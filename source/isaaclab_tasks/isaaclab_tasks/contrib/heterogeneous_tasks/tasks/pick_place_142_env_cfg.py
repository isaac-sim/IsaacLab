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
class PickPlace142EnvCfg(VLABenchmarkEnvCfg):
    """Place the green block into the white bowl, on a file cabinet."""

    instruction = "place the green block into the white bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.43437, -0.04382, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.44289, -0.02583, -0.0312), (0.0, 0.0, 0.909217, 0.416323)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.3064, 0.10713, -0.0075), (0.0, 0.0, -0.776977, 0.629529)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.466, 0.21677, -0.01471), (0.0, 0.0, 0.986247, 0.165276)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.yellow_block = library_object(
            "yellow_block", (0.3741, 0.19426, -0.0075), (0.0, 0.0, -0.016056, 0.999871)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("green_block")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
