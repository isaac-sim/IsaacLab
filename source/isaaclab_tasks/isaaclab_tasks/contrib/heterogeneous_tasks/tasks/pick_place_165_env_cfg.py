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
class PickPlace165EnvCfg(VLABenchmarkEnvCfg):
    """Move the marker into the black bowl, on a dining table."""

    instruction = "move the marker into the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.58561, 0.28758, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.42055, 0.16548, 0.07696), (0.0, 0.0, -0.893484, 0.449094)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.35286, -0.11828, 0.077), (0.0, 0.0, -0.926121, 0.377226)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cheez_it = library_object("cheez_it", (0.48848, -0.20695, 0.18374), (0.0, 0.0, 0.285557, 0.958362))
        self.scene.jello = library_object("jello", (0.54733, -0.02844, 0.09206), (0.0, 0.0, -0.401444, 0.915883))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.41627, -0.03145, 0.08644), (0.0, 0.0, 0.256999, 0.966412)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("dry_erase_marker")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
