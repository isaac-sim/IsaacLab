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
class PickPlace182EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the can of peaches and put it in the metal bin, on a storage unit."""

    instruction = "pick up the can of peaches and put it in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_halfunit = library_object(
            "standard_halfunit", (0.23901, -0.12223, -0.6376), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.43021, -0.1122, 0.156), (0.0, 0.0, -0.775089, 0.631852))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.fstylejug_a01 = library_object(
            "fstylejug_a01", (0.52558, 0.09945, 0.156), (0.0, 0.0, 0.295675, 0.955288)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.46438, 0.02598, 0.18489), (0.0, 0.0, -0.987323, 0.158726)
        )
        self.scene.wood_block = library_object(
            "wood_block", (0.33403, 0.05642, 0.25894), (0.0, 0.0, -0.983704, 0.179794)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_peaches")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
