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
class PickPlace126EnvCfg(VLABenchmarkEnvCfg):
    """Put the pumpkin and the kiwi in the metal bin, on a desk."""

    instruction = "put the pumpkin and the kiwi in the metal bin"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.44519, 0.61302, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.45048, -0.02748, 0.09), (0.0, 0.0, 0.825595, 0.564263))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.37847, 0.2227, 0.08999), (0.0, 0.0, 0.089953, 0.995946))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.30301, -0.0942, 0.08974), (0.0, 0.0, -0.977821, 0.209443)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.foam_roller = library_object(
            "foam_roller", (0.35054, -0.26538, 0.09005), (0.0, 0.0, -0.730186, 0.683248)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("pumpkinsmall"), SceneEntityCfg("av_kiwi01")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
