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
class PickPlace027EnvCfg(VLABenchmarkEnvCfg):
    """Place both the butter and the crab pen holder into the pan, on an end table."""

    instruction = "place both the butter and the crab pen holder into the pan"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.45119, -0.02125, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.38227, -0.13794, -0.05574), (0.0, 0.0, -0.999854, 0.017085))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.46967, 0.21744, -0.05603), (0.0, 0.0, -0.986742, 0.162298)
        )
        self.scene.butter = library_object("butter", (0.56939, -0.05935, -0.02959), (0.0, 0.0, -0.000255, 1.0))
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.53697, 0.11908, -0.01351), (0.0, 0.0, -0.008526, 0.999964)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.red_block = library_object("red_block", (0.29944, 0.18353, -0.0325), (0.0, 0.0, 0.86787, 0.496791))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("butter"), SceneEntityCfg("crabbypenholder")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
