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
class PickPlace134EnvCfg(VLABenchmarkEnvCfg):
    """Move the spoon into the wooden bowl, on a hexagonal table."""

    instruction = "move the spoon into the wooden bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.76429, -0.03094, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.41681, -0.08242, 0.07092), (0.0, 0.0, 0.99842, 0.056185)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.spaghetti = library_object("spaghetti", (0.36878, 0.209, 0.09592), (0.0, 0.0, -0.500766, 0.865583))
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.47106, 0.17707, 0.12202), (0.0, 0.0, 0.248775, 0.968561)
        )
        self.scene.spoon_2 = library_object("spoon_2", (0.54747, 0.18989, 0.07639), (0.0, 0.0, -0.706147, 0.708066))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_2")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
