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
class PickPlace226EnvCfg(VLABenchmarkEnvCfg):
    """Move the cleanser bottle into the decorative bowl, on a conference table."""

    instruction = "move the cleanser bottle into the decorative bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object(
            "steam", (0.76604, -0.38265, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.4187, 0.03053, 0.10904), (0.0, 0.0, 0.604375, 0.7967)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.48368, 0.21769, 0.16043), (0.0, 0.0, -0.48576, 0.874092)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.37926, 0.25345, 0.13789), (0.0, 0.0, -0.846784, 0.531937)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.33144, -0.15446, 0.10908), (0.0, 0.0, -0.938958, 0.344031)
        )
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.57003, -0.17565, 0.11844), (0.0, 0.0, -0.192443, 0.981308)
        )
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.43613, -0.236, 0.23432), (0.0, 0.0, -0.968291, 0.249825)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("soft_scrub")],
                "container": SceneEntityCfg("av_decorativebowl"),
                "require_gripper_detached": True,
            },
        )
