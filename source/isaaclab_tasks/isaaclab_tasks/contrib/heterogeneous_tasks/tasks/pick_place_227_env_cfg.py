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
class PickPlace227EnvCfg(VLABenchmarkEnvCfg):
    """Move the brick into the metal bin, on a table."""

    instruction = "move the brick into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.585, -0.18511, -0.69705), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.42569, -0.12383, 0.069), (0.0, 0.0, -0.655432, 0.755254))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.46229, 0.11169, 0.069), (0.0, 0.0, 0.523246, 0.852182)
        )
        self.scene.spatula_18 = library_object(
            "spatula_18", (0.40262, -0.24531, 0.07268), (0.0, 0.0, 0.057428, 0.99835)
        )
        self.scene.butter = library_object("butter", (0.28054, 0.19234, 0.09541), (0.0, 0.0, -0.995034, 0.099535))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.54236, -0.05368, 0.08334), (0.0, 0.0, 0.70506, 0.709147)
        )
        self.scene.brick = library_object("brick", (0.31471, 0.01356, 0.09457), (0.0, 0.0, -0.941517, 0.336967))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("brick")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
