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
class PickPlace239EnvCfg(VLABenchmarkEnvCfg):
    """Move the chess king into the pan, on a table."""

    instruction = "move the chess king into the pan"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.69592, -0.02642, -0.347), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.37621, -0.12773, 0.00926), (0.0, 0.0, -0.999925, 0.012253))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.5627, 0.01889, 0.09701), (0.0, 0.0, -0.013503, 0.999909))
        self.scene.oatmeal_raisin_cookies = library_object(
            "oatmeal_raisin_cookies", (0.43372, 0.23629, 0.09262), (0.0, 0.0, -0.957045, 0.289939)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.56119, -0.15241, 0.009), (0.0, 0.0, 0.326381, 0.945238)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.55712, 0.12629, 0.03811), (0.0, 0.0, 0.993403, 0.114673)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_king")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
