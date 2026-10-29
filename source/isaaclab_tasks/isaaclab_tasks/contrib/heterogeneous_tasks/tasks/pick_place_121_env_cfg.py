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
class PickPlace121EnvCfg(VLABenchmarkEnvCfg):
    """Set the can of peas and carrots upright on the plate, on a file cabinet."""

    instruction = "set the can of peas and carrots upright on the plate"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.50717, -0.12757, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.plate_small = library_object(
            "plate_small", (0.42126, 0.01753, -0.03109), (0.0, 0.0, -0.761462, 0.64821)
        )
        self.scene.av_kiwi01 = library_object(
            "av_kiwi01", (0.39449, 0.25902, -0.03101), (0.0, 0.0, -0.182306, 0.983242)
        )
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.33208, -0.26575, 0.02043), (0.0, 0.0, 0.703272, 0.710921)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.51691, -0.24124, -0.00174), (0.0, 0.0, -0.454555, 0.890719)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.4105, -0.2029, -0.031), (0.0, 0.0, -0.790327, 0.612685)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("peas_and_carrots")],
                "reference": SceneEntityCfg("plate_small"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
