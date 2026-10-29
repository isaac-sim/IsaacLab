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
class PickPlace191EnvCfg(VLABenchmarkEnvCfg):
    """Place the strawberry into the metal bin, on a table."""

    instruction = "place the strawberry into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_maple = library_object(
            "table_maple", (0.58509, -0.11722, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.52144, -0.03573, 0.009), (0.0, 0.0, 0.958108, 0.286407))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.29404, 0.04698, 0.00881), (0.0, 0.0, 0.999988, 0.004813)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.34172, 0.197, 0.071), (0.0, 0.0, -0.610825, 0.791766)
        )
        self.scene.av_knife = library_object("av_knife", (0.53401, 0.16881, 0.01006), (0.0, 0.0, 0.135959, 0.990715))
        self.scene.bagel = library_object("bagel", (0.40281, 0.07784, 0.02455), (0.0, 0.0, 0.657831, 0.753165))
        self.scene.birdhouse = library_object("birdhouse", (0.3618, -0.15767, 0.00905), (0.0, 0.0, 0.694552, 0.719442))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_stawberry")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
