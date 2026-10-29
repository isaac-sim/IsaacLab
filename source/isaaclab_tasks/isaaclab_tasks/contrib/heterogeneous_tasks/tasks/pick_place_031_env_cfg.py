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
class PickPlace031EnvCfg(VLABenchmarkEnvCfg):
    """Move the can of tuna into the salt bowl, on a dining table."""

    instruction = "move the can of tuna into the salt bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.60837, 0.34283, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.3721, 0.10591, 0.07106), (0.0, 0.0, -0.733547, 0.679639)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.4037, -0.13632, 0.08729), (0.0, 0.0, 0.90985, 0.414938)
        )
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.52847, -0.13739, 0.09703), (0.0, 0.0, -0.842476, 0.538734)
        )
        self.scene.mug_ycb = library_object("mug_ycb", (0.54057, 0.14385, 0.11169), (0.0, 0.0, 0.830011, 0.557747))
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.3249, -0.24647, 0.10011), (0.0, 0.0, -0.251688, 0.967808)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_tuna")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
