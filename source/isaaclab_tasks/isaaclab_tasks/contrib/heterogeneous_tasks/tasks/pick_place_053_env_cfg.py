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
class PickPlace053EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the tomato sauce can and put it in the pan, on a table."""

    instruction = "pick up the tomato sauce can and put it in the pan"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.55572, 0.1348, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.42569, -0.15682, 0.00926), (0.0, 0.0, 0.999735, 0.023024))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.31571, 0.19613, 0.05042), (0.0, 0.0, -0.590436, 0.807085)
        )
        self.scene.av_tincan = library_object("av_tincan", (0.50675, 0.21918, 0.009), (0.0, 0.0, -0.476111, 0.879385))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("tomato_sauce_can")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
