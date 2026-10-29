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
class PickPlace108EnvCfg(VLABenchmarkEnvCfg):
    """Move the can of corn into the metal bin, on an end table."""

    instruction = "move the can of corn into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.40829, -0.03374, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.41912, -0.02274, -0.001), (0.0, 0.0, -0.55659, 0.830788))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.29903, -0.08418, -0.001), (0.0, 0.0, -0.977233, 0.212167)
        )
        self.scene.corn_can = library_object("corn_can", (0.5498, 0.09288, 0.02802), (0.0, 0.0, 0.094985, 0.995479))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.30691, -0.17138, 0.07903), (0.0, 0.0, 0.269862, 0.962899)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("corn_can")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
