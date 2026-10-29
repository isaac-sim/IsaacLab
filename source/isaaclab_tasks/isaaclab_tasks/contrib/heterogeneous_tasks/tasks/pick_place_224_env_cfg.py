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
class PickPlace224EnvCfg(VLABenchmarkEnvCfg):
    """Put the milk jug in the metal bin, on a dining table."""

    instruction = "put the milk jug in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.53571, 0.81506, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.42951, 0.03873, 0.077), (0.0, 0.0, 0.429851, 0.9029))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.48278, -0.24764, 0.077), (0.0, 0.0, 0.0144, 0.999896)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.33615, -0.22707, 0.10384), (0.0, 0.0, 0.915158, 0.403096)
        )
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.33867, 0.2341, 0.077), (0.0, 0.0, -0.992735, 0.120325)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_eraser = library_object("av_eraser", (0.29424, -0.10792, 0.08133), (0.0, 0.0, 0.474892, 0.880044))
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.54734, -0.08076, 0.15703), (0.0, 0.0, 0.125575, 0.992084)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("milkjug_a01")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
