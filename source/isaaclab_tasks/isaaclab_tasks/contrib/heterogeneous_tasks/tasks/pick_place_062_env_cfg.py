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
class PickPlace062EnvCfg(VLABenchmarkEnvCfg):
    """Place the phone on top of the wooden tray, on a conference table."""

    instruction = "place the phone on top of the wooden tray"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object(
            "cline", (1.35651, -0.06325, -0.6971), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_woodentray_medium = library_object(
            "av_woodentray_medium", (0.4462, -0.04215, 0.129), (0.0, 0.0, 0.712469, 0.701703)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.42437, -0.27634, 0.1556), (0.0, 0.0, -0.671425, 0.741073)
        )
        self.scene.smartphone = library_object(
            "smartphone", (0.46094, 0.20472, 0.13465), (0.0, 0.0, 0.536649, 0.843805)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.32199, 0.20229, 0.15374), (0.0, 0.0, 0.999851, 0.017247)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("smartphone")],
                "reference": SceneEntityCfg("av_woodentray_medium"),
                "require_gripper_detached": True,
            },
        )
