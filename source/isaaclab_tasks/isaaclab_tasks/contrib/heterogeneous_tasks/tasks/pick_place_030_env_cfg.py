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
class PickPlace030EnvCfg(VLABenchmarkEnvCfg):
    """Put the barbecue sauce bottle in the metal bin, on a dining table."""

    instruction = "put the barbecue sauce bottle in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.61563, 0.38428, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.49489, -0.17854, 0.077), (0.0, 0.0, 0.744809, 0.667278))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.scissors = library_object("scissors", (0.37297, -0.07844, 0.08486), (0.0, 0.0, 0.999399, 0.034655))
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.32557, 0.14823, 0.07708), (0.0, 0.0, 0.412332, 0.911034)
        )
        self.scene.clamp = library_object("clamp", (0.47448, 0.18516, 0.09658), (0.0, 0.0, -0.991705, 0.128537))
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.56127, 0.01518, 0.07706), (0.0, 0.0, 0.656181, 0.754604)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
