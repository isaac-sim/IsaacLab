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
class PickPlace129EnvCfg(VLABenchmarkEnvCfg):
    """Take the spoon out of the metal bin, on a file cabinet."""

    instruction = "take the spoon out of the metal bin"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.516, 0.02593, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.43951, 0.0177, -0.031), (0.0, 0.0, -0.383028, 0.923737))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.mug = library_object("mug", (0.42253, -0.23296, -0.03082), (0.0, 0.0, 0.034064, 0.99942))
        self.scene.spoon = library_object("spoon", (0.43951, 0.0177, -0.00822), (0.0, 0.0, 0.992555, 0.1218))
        self.scene.mug_ycb = library_object("mug_ycb", (0.32828, 0.21957, 0.00969), (0.0, 0.0, -0.882168, 0.470934))
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.5475, 0.14132, -0.00189), (0.0, 0.0, 0.368464, 0.929642)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("spoon")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
