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
class PickPlace072EnvCfg(VLABenchmarkEnvCfg):
    """Put the sugar box on the sketchbook, on a console table."""

    instruction = "put the sugar box on the sketchbook"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.4112, -0.30854, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_sketchbook = library_object(
            "av_sketchbook", (0.39317, -0.01721, 0.20017), (0.0, 0.0, 0.757824, 0.652459)
        )
        self.scene.sugar_box = library_object("sugar_box", (0.36373, 0.21083, 0.28601), (0.0, 0.0, 0.009009, 0.999959))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.milkjug_a01 = library_object("milkjug_a01", (0.51756, 0.205, 0.198), (0.0, 0.0, 0.970233, 0.242173))
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.44758, -0.22728, 0.22403), (0.0, 0.0, -0.992456, 0.122601)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("sugar_box")],
                "reference": SceneEntityCfg("av_sketchbook"),
                "require_gripper_detached": True,
            },
        )
