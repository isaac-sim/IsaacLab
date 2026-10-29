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
class PickPlace007EnvCfg(VLABenchmarkEnvCfg):
    """Place the spoon into the pan, on a console table."""

    instruction = "place the spoon into the pan"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41098, 0.25704, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.42561, -0.12306, 0.06926), (0.0, 0.0, -0.999842, 0.017766))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.30107, 0.25065, 0.09789), (0.0, 0.0, -0.560793, 0.827956)
        )
        self.scene.spoon = library_object("spoon", (0.41787, 0.22705, 0.07778), (0.0, 0.0, -0.925828, 0.377945))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
