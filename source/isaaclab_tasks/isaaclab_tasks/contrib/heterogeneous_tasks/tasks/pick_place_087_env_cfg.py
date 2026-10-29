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
class PickPlace087EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the kiwi and put it in the wooden bowl, on a conference table."""

    instruction = "pick up the kiwi and put it in the wooden bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.gilbert = library_object(
            "gilbert", (0.72607, -0.11559, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.43594, 0.09715, 0.11892), (0.0, 0.0, 0.096411, 0.995342)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.3077, -0.12687, 0.11899), (0.0, 0.0, -0.254702, 0.96702))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.38598, -0.14585, 0.13529), (0.0, 0.0, 0.999836, 0.018111)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.5015, -0.16117, 0.11896), (0.0, 0.0, 0.670122, 0.742251)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.33581, -0.23261, 0.11906), (0.0, 0.0, 0.803837, 0.59485)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
