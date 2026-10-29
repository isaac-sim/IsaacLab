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
class PickPlace130EnvCfg(VLABenchmarkEnvCfg):
    """Place the blue block into the white bowl, on a dining table."""

    instruction = "place the blue block into the white bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.63804, -0.03401, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.45264, -0.04383, 0.0708), (0.0, 0.0, 0.868416, 0.495837)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.41761, -0.26734, 0.071), (0.0, 0.0, -0.713992, 0.700153)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.blue_block = library_object(
            "blue_block", (0.50376, -0.23801, 0.0945), (0.0, 0.0, 0.276743, 0.960944)
        )
        self.scene.clamp = library_object("clamp", (0.36524, 0.17312, 0.09058), (0.0, 0.0, -0.993106, 0.117218))
        self.scene.av_tincan = library_object("av_tincan", (0.52627, 0.19402, 0.071), (0.0, 0.0, -0.767307, 0.64128))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("blue_block")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
