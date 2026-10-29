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
class PickPlace000EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess pawn upright on the white tray, on a dining table."""

    instruction = "put the chess pawn upright on the white tray"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.65799, 0.19674, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_white_tray = library_object(
            "av_white_tray", (0.37227, 0.06462, 0.07106), (0.0, 0.0, -0.522537, 0.852617)
        )
        self.scene.spatula_04 = library_object("spatula_04", (0.52317, 0.12638, 0.07773), (0.0, 0.0, 0.861227, 0.50822))
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.35978, -0.11828, 0.071), (0.0, 0.0, 0.692404, 0.72151)
        )
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.55136, -0.0821, 0.11349), (0.0, 0.0, 0.360769, 0.932655)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.30816, -0.04352, 0.071), (0.0, 0.0, -0.596253, 0.802796)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.35389, -0.20447, 0.071), (0.0, 0.0, -0.977237, 0.212152)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.50815, -0.18363, 0.14107), (0.0, 0.0, 0.278894, 0.960322)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_chess_pawn")],
                "reference": SceneEntityCfg("av_white_tray"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
