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
class PickPlace220EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the chess queen and put it in the white bowl, on a dining table."""

    instruction = "pick up the chess queen and put it in the white bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.95148, -0.00571, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.40824, -0.12649, 0.0708), (0.0, 0.0, -0.996433, 0.084389)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_12 = library_object(
            "spatula_12", (0.41022, 0.06046, 0.07467), (0.0, 0.0, -0.998434, 0.05594)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.29198, 0.17427, 0.071), (0.0, 0.0, 0.619654, 0.784875)
        )
        self.scene.jello = library_object("jello", (0.38915, 0.21244, 0.08606), (0.0, 0.0, 0.148613, 0.988895))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
