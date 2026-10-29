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
class PickPlace167EnvCfg(VLABenchmarkEnvCfg):
    """Set the tomato soup can upright on the coaster, on an end table."""

    instruction = "set the tomato soup can upright on the coaster"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.47268, -0.03223, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.52198, 0.14158, -0.001), (0.0, 0.0, 0.69833, 0.715776))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.36072, -0.19092, -0.001), (0.0, 0.0, -0.657824, 0.753172)
        )
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.39411, -0.03657, -0.00103), (0.0, 0.0, -0.915034, 0.403377)
        )
        self.scene.fstylejug_a02 = library_object(
            "fstylejug_a02", (0.51634, -0.01718, -0.001), (0.0, 0.0, -0.678832, 0.734294)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.49902, -0.16945, 0.05002), (0.0, 0.0, 0.338149, 0.941092)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.37546, 0.08238, -0.001), (0.0, 0.0, -0.395821, 0.918328)
        )
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.28712, 0.00738, 0.07316), (0.0, 0.0, -0.988349, 0.152205)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("tomato_soup_can")],
                "reference": SceneEntityCfg("av_coaster_hexagon"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
