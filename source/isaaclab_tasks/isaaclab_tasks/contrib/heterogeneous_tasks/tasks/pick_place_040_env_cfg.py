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
class PickPlace040EnvCfg(VLABenchmarkEnvCfg):
    """Take the can of mushrooms out of the red bowl, on a file cabinet."""

    instruction = "take the can of mushrooms out of the red bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.43616, -0.00949, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.43928, 0.02015, -0.00352), (0.0, 0.0, 0.616361, 0.787463))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_09 = library_object(
            "spatula_09", (0.44897, -0.16101, -0.02427), (0.0, 0.0, 0.987982, 0.154572)
        )
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.40688, -0.25507, 0.02043), (0.0, 0.0, 0.991897, 0.127045)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.40937, 0.24625, -0.00211), (0.0, 0.0, -0.767803, 0.640686)
        )
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.43928, 0.02015, -0.00034), (0.0, 0.0, 0.6418, 0.766872)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.51632, 0.18779, -0.00221), (0.0, 0.0, 0.813789, 0.581161)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("canned_mushrooms")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
