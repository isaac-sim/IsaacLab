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
class PickPlace054EnvCfg(VLABenchmarkEnvCfg):
    """Move the frozen waffles into the black bowl, on a conference table."""

    instruction = "move the frozen waffles into the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object(
            "cline", (1.38624, -0.19455, -0.6971), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.39396, -0.04695, 0.12896), (0.0, 0.0, -0.970496, 0.241116)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.52783, 0.09948, 0.129), (0.0, 0.0, 0.688808, 0.724944)
        )
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.36069, 0.18449, 0.12898), (0.0, 0.0, -0.957949, 0.28694)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.raisin_box = library_object("raisin_box", (0.53089, -0.12755, 0.19059), (0.0, 0.0, 0.9154, 0.402545))
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.29799, 0.10965, 0.15826), (0.0, 0.0, -0.451156, 0.892445)
        )
        self.scene.mug_ycb = library_object("mug_ycb", (0.43048, -0.23792, 0.16969), (0.0, 0.0, -0.022019, 0.999758))
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.53356, 0.21955, 0.15811), (0.0, 0.0, 0.155157, 0.98789)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("frozen_waffles")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
