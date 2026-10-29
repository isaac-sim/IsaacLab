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
class PickPlace155EnvCfg(VLABenchmarkEnvCfg):
    """Place the mustard jar into the salt bowl, on a storage unit."""

    instruction = "place the mustard jar into the salt bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.6364, 0.11409, -0.6376), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.45667, 0.0244, 0.15606), (0.0, 0.0, -0.94134, 0.337458)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.fork_big = library_object("fork_big", (0.46854, -0.17755, 0.15635), (0.0, 0.0, 0.99789, 0.06493))
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.50764, -0.11162, 0.15608), (0.0, 0.0, 0.195035, 0.980796)
        )
        self.scene.bagel_00 = library_object("bagel_00", (0.32002, -0.08732, 0.17158), (0.0, 0.0, -0.620235, 0.784416))
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_mustard_jar")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
