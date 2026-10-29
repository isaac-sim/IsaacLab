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
class PickPlace076EnvCfg(VLABenchmarkEnvCfg):
    """Place the avocado into the metal bin, on a conference table."""

    instruction = "place the avocado into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object(
            "steam", (0.76679, -1.23583, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.44341, -0.07009, 0.109), (0.0, 0.0, 0.628214, 0.778041))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.spaghetti = library_object("spaghetti", (0.3403, 0.1093, 0.13392), (0.0, 0.0, 0.962857, 0.270011))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.31111, 0.21575, 0.10901), (0.0, 0.0, -0.914089, 0.405513)
        )
        self.scene.avocado01 = library_object("avocado01", (0.50734, 0.1862, 0.10883), (0.0, 0.0, -0.983866, 0.178907))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("avocado01")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
