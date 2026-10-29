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
class PickPlace012EnvCfg(VLABenchmarkEnvCfg):
    """Put the raisin box and the ranch dressing bottle in the metal bin, on a console table."""

    instruction = "put the raisin box and the ranch dressing bottle in the metal bin"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41578, -0.29277, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.39955, 0.07049, 0.069), (0.0, 0.0, -0.723459, 0.690368))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.29764, -0.22847, 0.14272), (0.0, 0.0, -0.652057, 0.75817)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.42013, -0.20091, 0.13059), (0.0, 0.0, -0.030306, 0.999541)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.51781, -0.16174, 0.09779), (0.0, 0.0, 0.305417, 0.952219)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("raisin_box"), SceneEntityCfg("ranch_dressing")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
