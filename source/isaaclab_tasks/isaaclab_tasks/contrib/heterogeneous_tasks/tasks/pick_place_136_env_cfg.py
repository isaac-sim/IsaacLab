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
class PickPlace136EnvCfg(VLABenchmarkEnvCfg):
    """Place the chess rook into the decorative bowl, on a sofa table."""

    instruction = "place the chess rook into the decorative bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_sofatable = library_object(
            "appleseed_sofatable", (0.43933, 0.25294, -0.69682), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.41247, -0.02346, 0.05804), (0.0, 0.0, -0.993414, 0.114583)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.40204, 0.20435, 0.058), (0.0, 0.0, -0.98288, 0.184245)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.4872, -0.22436, 0.08689), (0.0, 0.0, -0.655197, 0.755458)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.30102, 0.13129, 0.058), (0.0, 0.0, 0.359121, 0.933291)
        )
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.54445, 0.20763, 0.18332), (0.0, 0.0, -0.556176, 0.831064)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_rook")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
