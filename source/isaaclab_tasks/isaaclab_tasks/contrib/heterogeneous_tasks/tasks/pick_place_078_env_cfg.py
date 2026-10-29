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
class PickPlace078EnvCfg(VLABenchmarkEnvCfg):
    """Move the orange juice carton into the metal bin, on a table."""

    instruction = "move the orange juice carton into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.50562, 0.03154, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.46266, -0.05979, 0.009), (0.0, 0.0, 0.545956, 0.837814))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.57973, 0.12517, 0.08272), (0.0, 0.0, 0.033869, 0.999426)
        )
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.32849, 0.0686, 0.10524), (0.0, 0.0, -0.990915, 0.134487)
        )
        self.scene.measuring_cups = library_object(
            "measuring_cups", (0.42374, 0.23806, 0.02795), (0.0, 0.0, -0.126679, 0.991944)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.2934, -0.26066, 0.009), (0.0, 0.0, -0.524294, 0.851537)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_juice_carton")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
