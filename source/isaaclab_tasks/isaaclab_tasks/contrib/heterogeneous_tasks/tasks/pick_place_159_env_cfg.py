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
class PickPlace159EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the parmesan cheese and put it in the black bowl, on a hexagonal table."""

    instruction = "pick up the parmesan cheese and put it in the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.76049, 0.07402, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.36158, -0.05048, 0.07096), (0.0, 0.0, -0.740806, 0.671719)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_03 = library_object("spatula_03", (0.5551, -0.01615, 0.07773), (0.0, 0.0, 0.687093, 0.72657))
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.46674, 0.11538, 0.12243), (0.0, 0.0, -0.078567, 0.996909)
        )
        self.scene.milkjug_a02 = library_object(
            "milkjug_a02", (0.49437, -0.21871, 0.071), (0.0, 0.0, -0.26999, 0.962863)
        )
        self.scene.blue_block = library_object("blue_block", (0.30959, 0.115, 0.0945), (0.0, 0.0, -0.967739, 0.251956))
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.47644, 0.22696, 0.14495), (0.0, 0.0, 0.873395, 0.487012)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.37221, 0.23915, 0.14107), (0.0, 0.0, 0.655594, 0.755114)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("parmesan_cheese_canister")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
