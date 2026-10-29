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
class PickPlace210EnvCfg(VLABenchmarkEnvCfg):
    """Put the frozen vegetables in the metal bin, on a storage unit."""

    instruction = "put the frozen vegetables in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.24426, -0.20654, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.42986, 0.00392, 0.156), (0.0, 0.0, 0.583786, 0.811908))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.52038, 0.21257, 0.1795), (0.0, 0.0, -0.635061, 0.772462)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.524, -0.21053, 0.17229), (0.0, 0.0, 0.961577, 0.274534)
        )
        self.scene.av_apple = library_object("av_apple", (0.39266, -0.23874, 0.15696), (0.0, 0.0, -0.829194, 0.558961))
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.31279, 0.18974, 0.156), (0.0, 0.0, 0.090673, 0.995881)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("frozen_vegetable_block")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
