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
class PickPlace173EnvCfg(VLABenchmarkEnvCfg):
    """First put the yellow block in the decorative bowl, then put the pumpkin in it, on a table."""

    instruction = "first put the yellow block in the decorative bowl, then put the pumpkin in it"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.57505, -0.21137, -0.69705), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.44586, -0.04814, 0.06904), (0.0, 0.0, 0.685644, 0.727937)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.39827, 0.17947, 0.06874), (0.0, 0.0, -0.311019, 0.950404)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.yellow_block = library_object(
            "yellow_block", (0.41664, -0.22915, 0.0925), (0.0, 0.0, 0.954842, 0.297114)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.54983, 0.18459, 0.06906), (0.0, 0.0, -0.385369, 0.922762)
        )
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("yellow_block"), SceneEntityCfg("pumpkinsmall")],
                "container": SceneEntityCfg("av_decorativebowl"),
            },
        )
