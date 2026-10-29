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
class PickPlace089EnvCfg(VLABenchmarkEnvCfg):
    """Put the white bottle in the pan, on an office desk."""

    instruction = "put the white bottle in the pan"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.47359, 0.30969, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.45721, -0.00305, 0.00726), (0.0, 0.0, -0.997177, 0.075081))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.48193, -0.24039, 0.007), (0.0, 0.0, 0.612197, 0.790705)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.33901, -0.19227, 0.03579), (0.0, 0.0, -0.11097, 0.993824)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.5068, -0.16531, 0.08703), (0.0, 0.0, 0.621271, 0.783596)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a01")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
