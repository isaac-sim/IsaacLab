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
class PickPlace236EnvCfg(VLABenchmarkEnvCfg):
    """First put the chess rook in the metal bin, then put the chess pawn in it, on a storage unit."""

    instruction = "first put the chess rook in the metal bin, then put the chess pawn in it"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.55345, -0.20187, -0.6376), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.46638, -0.04677, 0.156), (0.0, 0.0, 0.329512, 0.944152))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_16 = library_object(
            "spatula_16", (0.33768, -0.01207, 0.15967), (0.0, 0.0, -0.119993, 0.992775)
        )
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.39273, -0.15868, 0.156), (0.0, 0.0, 0.911875, 0.410468)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.52506, 0.10769, 0.156), (0.0, 0.0, 0.998507, 0.054618)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("av_chess_rook"), SceneEntityCfg("av_chess_pawn")],
                "container": SceneEntityCfg("bin_a03"),
            },
        )
