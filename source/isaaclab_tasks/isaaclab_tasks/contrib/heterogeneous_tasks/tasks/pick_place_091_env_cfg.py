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
class PickPlace091EnvCfg(VLABenchmarkEnvCfg):
    """Move the chocolate pudding mix into the metal bin, on a table."""

    instruction = "move the chocolate pudding mix into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.91076, -0.00838, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.41139, -0.10333, -0.058), (0.0, 0.0, -0.629542, 0.776967))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.53029, 0.03993, -0.0345), (0.0, 0.0, -0.979049, 0.203623)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_plant_succulent_02 = library_object(
            "av_plant_succulent_02", (0.37518, 0.202, -0.058), (0.0, 0.0, -0.324064, 0.946035)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.5718, -0.03199, -0.03326), (0.0, 0.0, -0.220332, 0.975425)
        )
        self.scene.brick = library_object("brick", (0.49602, 0.12811, -0.03243), (0.0, 0.0, 0.517519, 0.855672))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
