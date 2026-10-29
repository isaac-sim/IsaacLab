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
class PickPlace234EnvCfg(VLABenchmarkEnvCfg):
    """Place the blue block on top of the coaster, on a desk."""

    instruction = "place the blue block on top of the coaster"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.50539, 0.19744, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.51047, 0.13392, 0.09), (0.0, 0.0, 0.807943, 0.58926))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.4659, -0.18495, 0.09), (0.0, 0.0, -0.080318, 0.996769)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.308, -0.15289, 0.1166), (0.0, 0.0, 0.19346, 0.981108)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.blue_block = library_object(
            "blue_block", (0.5622, -0.09704, 0.1135), (0.0, 0.0, -0.589495, 0.807772)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.36819, 0.23102, 0.09), (0.0, 0.0, -0.232978, 0.972482)
        )
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.34073, 0.09461, 0.21532), (0.0, 0.0, 0.851469, 0.524404)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("blue_block")],
                "reference": SceneEntityCfg("av_coaster_hexagon"),
                "require_gripper_detached": True,
            },
        )
