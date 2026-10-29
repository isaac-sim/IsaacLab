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
class PickPlace123EnvCfg(VLABenchmarkEnvCfg):
    """Place the pumpkin into the red bowl, on an office desk."""

    instruction = "place the pumpkin into the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.47688, 0.17457, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.43414, 0.20083, 0.03448), (0.0, 0.0, 0.676121, 0.73679))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.31431, 0.06726, 0.09501), (0.0, 0.0, 0.269962, 0.962871))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.42996, -0.07274, 0.00674), (0.0, 0.0, -0.837767, 0.546027)
        )
        self.scene.spoon_big = library_object("spoon_big", (0.33661, -0.1546, 0.00724), (0.0, 0.0, -0.548494, 0.836155))
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.53674, 0.08704, 0.007), (0.0, 0.0, 0.950114, 0.311902)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("pumpkinsmall")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
