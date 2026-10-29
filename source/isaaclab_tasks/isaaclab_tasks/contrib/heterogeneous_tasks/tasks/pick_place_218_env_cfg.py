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
class PickPlace218EnvCfg(VLABenchmarkEnvCfg):
    """Place the cream cheese into the salt bowl, on a console table."""

    instruction = "place the cream cheese into the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.43919, -0.37223, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.37109, 0.12436, 0.10906), (0.0, 0.0, -0.239454, 0.970908)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.4638, 0.27286, 0.1356), (0.0, 0.0, -0.783473, 0.621426)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.5146, -0.17039, 0.16002), (0.0, 0.0, 0.88935, 0.457226)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.2968, -0.0064, 0.10908), (0.0, 0.0, -0.596682, 0.802478)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.foam_roller = library_object(
            "foam_roller", (0.43554, -0.10467, 0.10905), (0.0, 0.0, -0.363909, 0.931435)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("cream_cheese")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
