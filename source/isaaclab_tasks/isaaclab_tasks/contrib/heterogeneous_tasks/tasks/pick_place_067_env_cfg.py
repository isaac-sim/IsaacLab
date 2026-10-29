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
class PickPlace067EnvCfg(VLABenchmarkEnvCfg):
    """Set the green block on the plate, on a file cabinet."""

    instruction = "set the green block on the plate"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.47457, -0.06285, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.plate_small = library_object(
            "plate_small", (0.41326, 0.04965, -0.03109), (0.0, 0.0, -0.717787, 0.696263)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.46174, 0.26223, -0.0075), (0.0, 0.0, 0.952818, 0.303542)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.bagel = library_object("bagel", (0.38285, -0.18792, -0.01545), (0.0, 0.0, -0.123256, 0.992375))
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.50622, -0.20637, -0.03094), (0.0, 0.0, 0.945915, 0.324416)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("green_block")],
                "reference": SceneEntityCfg("plate_small"),
                "require_gripper_detached": True,
            },
        )
