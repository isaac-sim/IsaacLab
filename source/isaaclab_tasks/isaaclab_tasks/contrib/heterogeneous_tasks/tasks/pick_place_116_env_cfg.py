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
class PickPlace116EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the white bottle and put it in the decorative bowl, on a conference table."""

    instruction = "pick up the white bottle and put it in the decorative bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object(
            "cline", (1.34497, -0.08278, -0.6971), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.42623, 0.11926, 0.12904), (0.0, 0.0, -0.02012, 0.999798)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_05 = library_object(
            "spatula_05", (0.41031, -0.07531, 0.13271), (0.0, 0.0, -0.999308, 0.037206)
        )
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.54634, -0.15815, 0.129), (0.0, 0.0, 0.952295, 0.305179)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.29752, -0.22109, 0.15789), (0.0, 0.0, -0.995628, 0.093406)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.49559, -0.2334, 0.21213), (0.0, 0.0, -0.905693, 0.423935)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a01")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
