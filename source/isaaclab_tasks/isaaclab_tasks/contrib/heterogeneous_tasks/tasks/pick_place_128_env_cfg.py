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
class PickPlace128EnvCfg(VLABenchmarkEnvCfg):
    """Place the popcorn box into the red bowl, on a storage unit."""

    instruction = "place the popcorn box into the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.22805, -0.20032, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.44673, -0.08885, 0.18348), (0.0, 0.0, -0.111568, 0.993757))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.56139, 0.02678, 0.19849), (0.0, 0.0, -0.969805, 0.243881)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.38077, -0.22123, 0.156), (0.0, 0.0, 0.174929, 0.984581)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("popcorn_box")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
