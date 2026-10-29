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
class PickPlace174EnvCfg(VLABenchmarkEnvCfg):
    """Set the avocado on the photo album, on a conference table."""

    instruction = "set the avocado on the photo album"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object(
            "cline", (1.37566, -0.07556, -0.6971), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.46607, 0.18391, 0.12906), (0.0, 0.0, 0.968851, 0.247644)
        )
        self.scene.av_photoalbum_closed = library_object(
            "av_photoalbum_closed", (0.44151, -0.03953, 0.129), (0.0, 0.0, -0.999915, 0.013022)
        )
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.28828, 0.04757, 0.12899), (0.0, 0.0, -0.014663, 0.999892))
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.28963, -0.09085, 0.129), (0.0, 0.0, 0.032566, 0.99947)
        )
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.39519, -0.20605, 0.15781), (0.0, 0.0, 0.846344, 0.532637)
        )
        self.scene.spoon_big = library_object("spoon_big", (0.576, -0.05672, 0.12924), (0.0, 0.0, -0.721567, 0.692345))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.avocado01 = library_object("avocado01", (0.29975, 0.15349, 0.12883), (0.0, 0.0, -0.338997, 0.940788))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("avocado01")],
                "reference": SceneEntityCfg("av_photoalbum_closed"),
                "require_gripper_detached": True,
            },
        )
