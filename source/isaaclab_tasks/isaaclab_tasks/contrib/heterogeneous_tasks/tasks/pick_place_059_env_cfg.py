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
class PickPlace059EnvCfg(VLABenchmarkEnvCfg):
    """Move the orange can into the metal bin, on an end table."""

    instruction = "move the orange can into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.42964, 0.00961, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.44318, -0.12534, -0.001), (0.0, 0.0, -0.972527, 0.232789))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.soup_can = library_object("soup_can", (0.49982, 0.12276, -0.00094), (0.0, 0.0, -0.911281, 0.411786))
        self.scene.spoon_big = library_object(
            "spoon_big", (0.40669, 0.01256, -0.00076), (0.0, 0.0, -0.981113, 0.193436)
        )
        self.scene.bagel = library_object("bagel", (0.32272, 0.1895, 0.01455), (0.0, 0.0, 0.999851, 0.017238))
        self.scene.red_block = library_object("red_block", (0.4327, 0.1896, 0.0225), (0.0, 0.0, 0.64682, 0.762643))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("soup_can")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
