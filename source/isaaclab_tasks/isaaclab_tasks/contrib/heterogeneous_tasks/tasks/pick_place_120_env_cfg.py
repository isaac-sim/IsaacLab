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
class PickPlace120EnvCfg(VLABenchmarkEnvCfg):
    """Set the brick on the sketchbook, on a conference table."""

    instruction = "set the brick on the sketchbook"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.68776, 0.59843, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_sketchbook = library_object(
            "av_sketchbook", (0.42613, 0.11578, 0.13117), (0.0, 0.0, -0.414299, 0.910141)
        )
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.4153, -0.19044, 0.12899), (0.0, 0.0, 0.914255, 0.405139))
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.29463, -0.06285, 0.20272), (0.0, 0.0, -0.23131, 0.97288)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.33159, -0.14215, 0.191), (0.0, 0.0, -0.383622, 0.92349)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.55111, -0.06993, 0.22418), (0.0, 0.0, 0.366398, 0.930458)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.46914, -0.13132, 0.20295), (0.0, 0.0, -0.735134, 0.677922)
        )
        self.scene.brick = library_object("brick", (0.48668, -0.23873, 0.15457), (0.0, 0.0, -0.174088, 0.98473))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("brick")],
                "reference": SceneEntityCfg("av_sketchbook"),
                "require_gripper_detached": True,
            },
        )
