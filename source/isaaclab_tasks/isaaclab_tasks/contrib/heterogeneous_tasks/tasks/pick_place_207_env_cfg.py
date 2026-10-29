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
class PickPlace207EnvCfg(VLABenchmarkEnvCfg):
    """Put the kiwi in the white bowl, on a desk."""

    instruction = "put the kiwi in the white bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.47495, 0.04467, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.426, -0.02492, 0.0898), (0.0, 0.0, 0.980364, 0.197198)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.40925, 0.21332, 0.08999), (0.0, 0.0, 0.276302, 0.961071))
        self.scene.fstylejug_a01 = library_object(
            "fstylejug_a01", (0.4948, -0.22147, 0.09), (0.0, 0.0, -0.603039, 0.797712)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.49256, 0.13477, 0.14102), (0.0, 0.0, -0.096399, 0.995343)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.brick = library_object("brick", (0.31778, -0.19404, 0.11557), (0.0, 0.0, -0.712292, 0.701883))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
