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
class PickPlace225EnvCfg(VLABenchmarkEnvCfg):
    """Stack the Rubik's cube on the orange can, on a dining table."""

    instruction = "stack the Rubik's cube on the orange can"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.54644, 0.84858, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.36366, -0.12359, 0.07125), (0.0, 0.0, 0.441523, 0.89725)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.29509, 0.00306, 0.071), (0.0, 0.0, -0.888226, 0.459407)
        )
        self.scene.soup_can = library_object("soup_can", (0.48495, 0.11901, 0.07106), (0.0, 0.0, -0.400866, 0.916137))
        self.scene.storage_box = library_object(
            "storage_box", (0.51121, 0.21468, 0.07087), (0.0, 0.0, 0.642703, 0.766115)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.3635, 0.1887, 0.10239), (0.0, 0.0, 0.795004, 0.606604)
        )
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.3957, 0.23153, 0.14516), (0.0, 0.0, 0.278015, 0.960577)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("rubiks_cube")],
                "reference": SceneEntityCfg("soup_can"),
                "require_gripper_detached": True,
            },
        )
