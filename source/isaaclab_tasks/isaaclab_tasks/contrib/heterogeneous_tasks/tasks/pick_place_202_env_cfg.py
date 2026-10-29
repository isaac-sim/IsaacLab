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
class PickPlace202EnvCfg(VLABenchmarkEnvCfg):
    """Place both the milk jug and the mustard bottle into the pan, on a file cabinet."""

    instruction = "place both the milk jug and the mustard bottle into the pan"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.42626, 0.11989, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.4484, 0.09931, -0.03074), (0.0, 0.0, -0.055355, 0.998467))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.38603, -0.2436, -0.031), (0.0, 0.0, 0.885436, 0.464762)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.51614, 0.26251, 0.04903), (0.0, 0.0, 0.999988, 0.004827)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("milkjug_a01"), SceneEntityCfg("mustard_bottle")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
