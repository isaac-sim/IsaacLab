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
class PickPlace105EnvCfg(VLABenchmarkEnvCfg):
    """Remove the chocolate pudding mix from the red bowl, on a table."""

    instruction = "remove the chocolate pudding mix from the red bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.50539, 0.07847, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.37305, 0.0048, 0.03648), (0.0, 0.0, 0.987915, 0.154998))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_08 = library_object(
            "spatula_08", (0.51685, 0.09007, 0.01573), (0.0, 0.0, -0.502943, 0.86432)
        )
        self.scene.butter = library_object("butter", (0.31884, -0.27303, 0.03541), (0.0, 0.0, 0.654259, 0.756271))
        self.scene.granola_bars = library_object(
            "granola_bars", (0.42623, 0.26486, 0.071), (0.0, 0.0, -0.700305, 0.713844)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.49262, -0.13335, 0.00908), (0.0, 0.0, -0.015678, 0.999877)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.37305, 0.0048, 0.04774), (0.0, 0.0, -0.240436, 0.970665)
        )
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.46983, -0.22959, 0.13432), (0.0, 0.0, -0.051269, 0.998685)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
