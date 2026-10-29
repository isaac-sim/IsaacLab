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
class PickPlace244EnvCfg(VLABenchmarkEnvCfg):
    """Set the ranch dressing bottle upright on the white tray, on an office desk."""

    instruction = "set the ranch dressing bottle upright on the white tray"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.53704, 0.36658, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_white_tray = library_object(
            "av_white_tray", (0.40232, 0.12558, 0.00706), (0.0, 0.0, -0.651443, 0.758698)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.51801, 0.05927, 0.08072), (0.0, 0.0, -0.589355, 0.807874)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.29897, 0.12796, 0.06859), (0.0, 0.0, -0.276749, 0.960942)
        )
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.54174, -0.16756, 0.0813), (0.0, 0.0, -0.84125, 0.540646)
        )
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.50765, 0.21584, 0.02134), (0.0, 0.0, -0.570069, 0.821597)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.36045, -0.15986, 0.07707), (0.0, 0.0, 0.751397, 0.659851)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("ranch_dressing")],
                "reference": SceneEntityCfg("av_white_tray"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
