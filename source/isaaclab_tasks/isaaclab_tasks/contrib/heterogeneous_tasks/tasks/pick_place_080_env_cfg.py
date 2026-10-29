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
class PickPlace080EnvCfg(VLABenchmarkEnvCfg):
    """Take the red block out of the metal bin, on a table."""

    instruction = "take the red block out of the metal bin"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.51508, 0.18044, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.41102, 0.17982, 0.009), (0.0, 0.0, -0.00478, 0.999989))
        self.scene.av_book_77 = library_object("av_book_77", (0.3953, -0.08076, 0.009), (0.0, 0.0, 0.124368, 0.992236))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.29238, 0.18581, 0.08272), (0.0, 0.0, 0.71938, 0.694617)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.57229, 0.03312, 0.03789), (0.0, 0.0, -0.053623, 0.998561)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.55639, 0.11756, 0.03779), (0.0, 0.0, 0.998072, 0.062068)
        )
        self.scene.red_block = library_object("red_block", (0.41102, 0.17982, 0.0465), (0.0, 0.0, -0.986777, 0.162086))
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("red_block")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
