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
class PickPlace092EnvCfg(VLABenchmarkEnvCfg):
    """Move the foam roller into the decorative bowl, on a table."""

    instruction = "move the foam roller into the decorative bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.93459, -0.03902, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.43027, -0.08924, -0.05796), (0.0, 0.0, 0.66032, 0.750984)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.5104, 0.14762, -0.058), (0.0, 0.0, 0.780985, 0.62455)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.31279, 0.12274, -0.02911), (0.0, 0.0, 0.208897, 0.977938)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.4391, 0.09611, -0.058), (0.0, 0.0, 0.17286, 0.984946)
        )
        self.scene.foam_roller = library_object(
            "foam_roller", (0.42093, 0.20708, -0.05795), (0.0, 0.0, -0.671484, 0.741019)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("foam_roller")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
