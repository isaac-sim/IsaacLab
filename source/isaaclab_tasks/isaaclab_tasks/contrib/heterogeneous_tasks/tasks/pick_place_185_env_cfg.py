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
class PickPlace185EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the raisin box and put it in the metal bin, on an office desk."""

    instruction = "pick up the raisin box and put it in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.51415, -0.1568, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.38529, 0.04575, 0.007), (0.0, 0.0, 0.743142, 0.669134))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.raisin_box = library_object(
            "raisin_box", (0.40419, -0.21123, 0.06859), (0.0, 0.0, -0.215848, 0.976427)
        )
        self.scene.blue_block = library_object(
            "blue_block", (0.48527, -0.24591, 0.0305), (0.0, 0.0, -0.612206, 0.790698)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.av_tincan = library_object("av_tincan", (0.55945, 0.03019, 0.007), (0.0, 0.0, 0.246362, 0.969178))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("raisin_box")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
