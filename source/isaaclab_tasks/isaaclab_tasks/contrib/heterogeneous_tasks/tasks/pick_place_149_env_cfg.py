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
class PickPlace149EnvCfg(VLABenchmarkEnvCfg):
    """Put the white bottle in the metal bowl, on a storage cabinet."""

    instruction = "put the white bottle in the metal bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.46728, 0.05802, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.466, 0.11493, -0.03075), (0.0, 0.0, 0.617641, 0.78646)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.33906, 0.19534, -0.03103), (0.0, 0.0, -0.984144, 0.177373)
        )
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.44421, -0.20283, -0.031), (0.0, 0.0, -0.882614, 0.470098)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.30497, -0.17154, -0.01424), (0.0, 0.0, 0.695474, 0.718551))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.35042, -0.03522, -0.03104), (0.0, 0.0, -0.363974, 0.931409)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.54194, -0.0626, 0.03907), (0.0, 0.0, 0.809075, 0.587706)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
