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
class PickPlace170EnvCfg(VLABenchmarkEnvCfg):
    """First put the chocolate pudding mix in the red bowl, then put the can of corn in it, on a conference table."""

    instruction = "first put the chocolate pudding mix in the red bowl, then put the can of corn in it"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.gilbert = library_object(
            "gilbert", (0.78623, -0.08952, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.41571, -0.15894, 0.14648), (0.0, 0.0, 0.855088, 0.518483))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.fstylejug_a01 = library_object(
            "fstylejug_a01", (0.3057, 0.04982, 0.119), (0.0, 0.0, -0.978385, 0.20679)
        )
        self.scene.corn_can = library_object("corn_can", (0.57053, 0.05538, 0.14802), (0.0, 0.0, 0.675905, 0.736989))
        self.scene.spoon = library_object("spoon", (0.45087, 0.05932, 0.12778), (0.0, 0.0, 0.498237, 0.867041))
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.3857, 0.25459, 0.14374), (0.0, 0.0, -0.901011, 0.433797)
        )
        self.scene.mustard = library_object("mustard", (0.38074, 0.15673, 0.2147), (0.0, 0.0, -0.87201, 0.489488))
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix"), SceneEntityCfg("corn_can")],
                "container": SceneEntityCfg("bowl_ycb"),
            },
        )
