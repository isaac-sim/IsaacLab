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
class PickPlace243EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess queen in the metal bin, on a table."""

    instruction = "put the chess queen in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.52518, -0.3767, -0.69705), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.47504, 0.18097, 0.069), (0.0, 0.0, -0.852138, 0.523317))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_19 = library_object(
            "spatula_19", (0.45387, -0.12523, 0.07268), (0.0, 0.0, -0.127026, 0.991899)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.30406, 0.00625, 0.069), (0.0, 0.0, 0.683977, 0.729503)
        )
        self.scene.mug = library_object("mug", (0.35781, -0.22736, 0.06918), (0.0, 0.0, -0.424583, 0.905389))
        self.scene.spoon_2 = library_object("spoon_2", (0.32742, 0.16912, 0.07439), (0.0, 0.0, -0.314961, 0.949105))
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.41642, 0.00176, 0.08334), (0.0, 0.0, 0.916284, 0.400528)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
