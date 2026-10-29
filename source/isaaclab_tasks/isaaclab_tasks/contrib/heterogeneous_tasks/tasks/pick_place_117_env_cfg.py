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
class PickPlace117EnvCfg(VLABenchmarkEnvCfg):
    """Put the chocolate pudding mix in the metal bowl, on a dining table."""

    instruction = "put the chocolate pudding mix in the metal bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (0.61452, 0.2904, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.38654, 0.03568, 0.07125), (0.0, 0.0, 0.346425, 0.938078)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.36959, -0.21777, 0.071), (0.0, 0.0, 0.073827, 0.997271)
        )
        self.scene.spatula_13 = library_object("spatula_13", (0.54549, 0.08231, 0.07467), (0.0, 0.0, 0.16225, 0.98675))
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.50391, -0.24024, 0.071), (0.0, 0.0, -0.326256, 0.945281)
        )
        self.scene.raisin_box = library_object("raisin_box", (0.38886, 0.23321, 0.13259), (0.0, 0.0, 0.71968, 0.694306))
        self.scene.storage_box = library_object(
            "storage_box", (0.48057, -0.14051, 0.07087), (0.0, 0.0, 0.440348, 0.897827)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.32251, -0.06887, 0.09574), (0.0, 0.0, 0.307173, 0.951654)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
