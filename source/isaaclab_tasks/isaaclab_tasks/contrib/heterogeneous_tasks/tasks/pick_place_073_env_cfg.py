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
class PickPlace073EnvCfg(VLABenchmarkEnvCfg):
    """Put the can of peaches in the metal bin, on a conference table."""

    instruction = "put the can of peaches in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.gilbert = library_object(
            "gilbert", (0.73573, 0.56783, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.36519, -0.00413, 0.119), (0.0, 0.0, 0.695233, 0.718784))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object("orange_02", (0.45693, -0.24934, 0.11906), (0.0, 0.0, 0.462851, 0.886436))
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.54624, 0.12934, 0.14789), (0.0, 0.0, -0.072138, 0.997395)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.58066, -0.07619, 0.11901), (0.0, 0.0, -0.999088, 0.042709)
        )
        self.scene.mustard = library_object("mustard", (0.39175, 0.25619, 0.2147), (0.0, 0.0, 0.054936, 0.99849))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_peaches")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
