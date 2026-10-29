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
class PickPlace055EnvCfg(VLABenchmarkEnvCfg):
    """Place the crab pen holder into the pan, on an office desk."""

    instruction = "place the crab pen holder into the pan"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.51374, 0.03644, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.40934, -0.09374, 0.00726), (0.0, 0.0, -0.999805, 0.019725))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.41527, 0.27318, 0.00697), (0.0, 0.0, -0.980806, 0.194985)
        )
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.50748, -0.26513, 0.007), (0.0, 0.0, 0.198312, 0.980139)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.31078, 0.22092, 0.06859), (0.0, 0.0, -0.640976, 0.767561)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("crabbypenholder")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
