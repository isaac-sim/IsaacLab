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
class PickPlace140EnvCfg(VLABenchmarkEnvCfg):
    """Move the pumpkin into the metal bowl, on an end table."""

    instruction = "move the pumpkin into the metal bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.43062, 0.02995, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.44166, 0.05536, -0.00075), (0.0, 0.0, 0.998438, 0.055867)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.54102, 0.21227, 0.07272), (0.0, 0.0, 0.019783, 0.999804)
        )
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.35487, -0.1662, -0.00126), (0.0, 0.0, -0.98127, 0.192636)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.52394, -0.10089, -0.001), (0.0, 0.0, 0.605031, 0.796202)
        )
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.32564, 0.1966, -0.00099), (0.0, 0.0, 0.082196, 0.996616)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("pumpkinsmall")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
