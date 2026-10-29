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
class PickPlace056EnvCfg(VLABenchmarkEnvCfg):
    """Place the orange can upright on the cardboard box, on a console table."""

    instruction = "place the orange can upright on the cardboard box"
    episode_length_s = 80.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.43567, 0.18642, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_08 = library_object(
            "spatula_08", (0.32163, -0.08157, 0.11573), (0.0, 0.0, 0.809001, 0.587807)
        )
        self.scene.cubebox_a02 = library_object(
            "cubebox_a02", (0.48359, -0.07519, 0.109), (0.0, 0.0, 0.724073, 0.689724)
        )
        self.scene.soup_can = library_object("soup_can", (0.41769, 0.05744, 0.10906), (0.0, 0.0, 0.004406, 0.99999))
        self.scene.av_knife = library_object("av_knife", (0.42728, 0.22333, 0.11006), (0.0, 0.0, 0.456251, 0.889851))
        self.scene.yellow_block = library_object(
            "yellow_block", (0.51746, -0.2326, 0.1325), (0.0, 0.0, 0.067256, 0.997736)
        )
        self.scene.potato_masher = library_object(
            "potato_masher", (0.53613, 0.10835, 0.10942), (0.0, 0.0, 0.813306, 0.581836)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("soup_can")],
                "reference": SceneEntityCfg("cubebox_a02"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
