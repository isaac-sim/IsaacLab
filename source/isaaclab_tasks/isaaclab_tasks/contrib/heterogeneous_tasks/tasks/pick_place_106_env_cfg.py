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
class PickPlace106EnvCfg(VLABenchmarkEnvCfg):
    """Place the wooden candle holder into the metal bin, on a table."""

    instruction = "place the wooden candle holder into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_maple = library_object(
            "table_maple", (0.47549, 0.13283, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.43353, 0.08645, 0.009), (0.0, 0.0, 0.270274, 0.962783))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.38216, -0.13468, 0.009), (0.0, 0.0, -0.127162, 0.991882)
        )
        self.scene.fork_big = library_object("fork_big", (0.43719, -0.26775, 0.00935), (0.0, 0.0, 0.058515, 0.998287))
        self.scene.blue_block = library_object(
            "blue_block", (0.47586, -0.15225, 0.0325), (0.0, 0.0, -0.442387, 0.896824)
        )
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.37549, 0.22476, 0.00901), (0.0, 0.0, 0.414528, 0.910036)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_woodenchmedium")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
