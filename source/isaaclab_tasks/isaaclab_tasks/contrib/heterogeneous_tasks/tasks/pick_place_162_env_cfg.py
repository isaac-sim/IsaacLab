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
class PickPlace162EnvCfg(VLABenchmarkEnvCfg):
    """Place the milk jug into the wooden bowl, on an end table."""

    instruction = "place the milk jug into the wooden bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.44584, -0.01395, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.41362, -0.05534, -0.05608), (0.0, 0.0, -0.717352, 0.696711)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object(
            "av_kiwi01", (0.30805, 0.22313, -0.05601), (0.0, 0.0, -0.999782, 0.020877)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.5105, 0.17377, -0.056), (0.0, 0.0, -0.930167, 0.367136)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.31552, 0.16904, -0.02461), (0.0, 0.0, -0.968168, 0.2503)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("milkjug_a01")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
