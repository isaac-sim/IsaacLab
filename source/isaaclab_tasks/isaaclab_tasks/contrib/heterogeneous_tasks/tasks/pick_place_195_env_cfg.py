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
class PickPlace195EnvCfg(VLABenchmarkEnvCfg):
    """Put the spoon in the metal bowl, on a console table."""

    instruction = "put the spoon in the metal bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.41159, 0.28271, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.39388, -0.1355, 0.19825), (0.0, 0.0, 0.682619, 0.730775)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.27674, -0.09423, 0.198), (0.0, 0.0, 0.503986, 0.863712)
        )
        self.scene.mug = library_object("mug", (0.33694, 0.16602, 0.19818), (0.0, 0.0, 0.956175, 0.292794))
        self.scene.spoon_2 = library_object("spoon_2", (0.48577, 0.07921, 0.20339), (0.0, 0.0, 0.808635, 0.588311))
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.53731, -0.02969, 0.198), (0.0, 0.0, -0.407573, 0.913172)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.31406, 0.05207, 0.22711), (0.0, 0.0, 0.02582, 0.999667)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_2")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
