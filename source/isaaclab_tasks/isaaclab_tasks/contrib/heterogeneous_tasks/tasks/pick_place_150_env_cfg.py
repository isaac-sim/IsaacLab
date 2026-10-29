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
class PickPlace150EnvCfg(VLABenchmarkEnvCfg):
    """Remove the mustard bottle from the metal bin, on an office desk."""

    instruction = "remove the mustard bottle from the metal bin"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.51377, -0.25512, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.43219, -0.09908, 0.007), (0.0, 0.0, -0.885638, 0.464377))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object("orange_02", (0.32619, 0.09167, 0.00706), (0.0, 0.0, -0.692012, 0.721886))
        self.scene.av_hippoorganizer = library_object(
            "av_hippoorganizer", (0.50689, 0.20959, 0.007), (0.0, 0.0, -0.99993, 0.011817)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.32762, 0.20107, 0.10218), (0.0, 0.0, -0.939136, 0.343546)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard = library_object("mustard", (0.43219, -0.09908, 0.1167), (0.0, 0.0, 0.238873, 0.971051))
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("mustard")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
