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
class PickPlace052EnvCfg(VLABenchmarkEnvCfg):
    """Put a bottle in the red bowl, on a console table."""

    instruction = "put a bottle in the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41264, 0.11423, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.40256, -0.21126, 0.09648), (0.0, 0.0, 0.73855, 0.674199))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.50107, -0.03951, 0.09584), (0.0, 0.0, 0.986204, 0.165533)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.37354, 0.00216, 0.131), (0.0, 0.0, 0.483635, 0.87527)
        )
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.40197, 0.08475, 0.1433), (0.0, 0.0, 0.629053, 0.777362)
        )
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.32391, -0.07399, 0.069), (0.0, 0.0, -0.600896, 0.799327)
        )
        self.scene.red_block = library_object("red_block", (0.50082, 0.21034, 0.0925), (0.0, 0.0, -0.915277, 0.402825))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("ketchup_bottle"), SceneEntityCfg("naturalbostonroundbottle_a02")],
                "container": SceneEntityCfg("bowl_ycb"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
