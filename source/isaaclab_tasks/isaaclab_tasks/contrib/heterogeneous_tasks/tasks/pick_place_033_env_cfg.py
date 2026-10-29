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
class PickPlace033EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the marker and put it in the metal bin, on a desk."""

    instruction = "pick up the marker and put it in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.47467, 0.12226, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.39415, 0.00336, 0.09), (0.0, 0.0, 0.686186, 0.727426))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.51074, -0.2399, 0.09), (0.0, 0.0, 0.088248, 0.996098)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.45749, 0.25263, 0.09944), (0.0, 0.0, 0.561361, 0.827571)
        )
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.55773, 0.0162, 0.16416), (0.0, 0.0, 0.080598, 0.996747)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("dry_erase_marker")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
