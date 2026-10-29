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
class PickPlace124EnvCfg(VLABenchmarkEnvCfg):
    """Move the barbecue sauce bottle into the pan, on a table."""

    instruction = "move the barbecue sauce bottle into the pan"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.51666, 0.13414, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_pan_b = library_object("av_pan_b", (0.4089, -0.07586, 0.00888), (0.0, 0.0, 0.999334, 0.036488))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.3358, -0.26343, 0.00906), (0.0, 0.0, -0.779793, 0.626038)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                "container": SceneEntityCfg("av_pan_b"),
                "require_gripper_detached": True,
            },
        )
