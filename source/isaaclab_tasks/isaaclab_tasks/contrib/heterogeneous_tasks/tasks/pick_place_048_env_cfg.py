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
class PickPlace048EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the milk carton and put it in the salt bowl, on a dining table."""

    instruction = "pick up the milk carton and put it in the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (1.38683, -0.03695, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.49532, -0.00941, 0.07706), (0.0, 0.0, -0.086761, 0.996229)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.ceramic_mug = library_object(
            "ceramic_mug", (0.32905, -0.0124, 0.07698), (0.0, 0.0, 0.975591, 0.219597)
        )
        self.scene.spaghetti = library_object("spaghetti", (0.42878, -0.15262, 0.10192), (0.0, 0.0, 0.686229, 0.727386))
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.37336, -0.23012, 0.10581), (0.0, 0.0, 0.348845, 0.93718)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.40849, 0.21023, 0.17218), (0.0, 0.0, 0.751092, 0.660198)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.55606, 0.17121, 0.15095), (0.0, 0.0, -0.524599, 0.85135)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("milk_carton")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
