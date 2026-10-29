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
class PickPlace008EnvCfg(VLABenchmarkEnvCfg):
    """Place both the can of peaches and the can of green beans into the decorative bowl, on a hexagonal table."""

    instruction = "place both the can of peaches and the can of green beans into the decorative bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.74801, -0.04179, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.39533, 0.0194, 0.07104), (0.0, 0.0, 0.692445, 0.721471)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.35459, 0.22595, 0.09989), (0.0, 0.0, 0.050577, 0.99872)
        )
        self.scene.measuring_cups = library_object(
            "measuring_cups", (0.34335, -0.22576, 0.08995), (0.0, 0.0, -0.999622, 0.027498)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.53765, 0.22625, 0.09979), (0.0, 0.0, 0.844433, 0.535661)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_peaches"), SceneEntityCfg("green_beans_can")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
