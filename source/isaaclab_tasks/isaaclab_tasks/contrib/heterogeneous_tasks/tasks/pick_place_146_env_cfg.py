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
class PickPlace146EnvCfg(VLABenchmarkEnvCfg):
    """Place the Rubik's cube on top of the white tray, on a table."""

    instruction = "place the Rubik's cube on top of the white tray"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.55582, -0.14855, -0.69705), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_white_tray = library_object(
            "av_white_tray", (0.43209, -0.12956, 0.06906), (0.0, 0.0, -0.826687, 0.562663)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.56594, 0.18235, 0.06881), (0.0, 0.0, -0.710917, 0.703276)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.54394, -0.14172, 0.08576), (0.0, 0.0, -0.21088, 0.977512))
        self.scene.spoon = library_object("spoon", (0.45692, 0.18719, 0.07778), (0.0, 0.0, -0.965303, 0.261131))
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.35385, 0.2031, 0.10039), (0.0, 0.0, -0.302307, 0.953211)
        )
        self.scene.chocolate_pudding = library_object(
            "chocolate_pudding", (0.33376, 0.06828, 0.08841), (0.0, 0.0, 0.756666, 0.653802)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("rubiks_cube")],
                "reference": SceneEntityCfg("av_white_tray"),
                "require_gripper_detached": True,
            },
        )
