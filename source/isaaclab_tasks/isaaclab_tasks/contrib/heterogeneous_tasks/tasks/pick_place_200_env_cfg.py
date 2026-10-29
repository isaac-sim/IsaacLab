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
class PickPlace200EnvCfg(VLABenchmarkEnvCfg):
    """Remove the blue block from the red bowl, on a table."""

    instruction = "remove the blue block from the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.87521, -0.05895, -0.69705), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.486, 0.09173, 0.09648), (0.0, 0.0, 0.775222, 0.631689))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_07 = library_object(
            "spatula_07", (0.42554, -0.19632, 0.07214), (0.0, 0.0, 0.221626, 0.975132)
        )
        self.scene.butter = library_object("butter", (0.46123, 0.23319, 0.09541), (0.0, 0.0, -0.39592, 0.918285))
        self.scene.smartphone = library_object(
            "smartphone", (0.32796, 0.07536, 0.07465), (0.0, 0.0, 0.999934, 0.011499)
        )
        self.scene.blue_block = library_object("blue_block", (0.486, 0.09173, 0.1065), (0.0, 0.0, 0.486207, 0.873844))
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.34463, -0.0556, 0.06901), (0.0, 0.0, -0.584522, 0.811378)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.30474, 0.24046, 0.09811), (0.0, 0.0, -0.642813, 0.766023)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("blue_block")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
