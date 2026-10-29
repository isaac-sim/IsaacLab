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
class PickPlace109EnvCfg(VLABenchmarkEnvCfg):
    """Move the can of peaches into the black bowl, on a conference table."""

    instruction = "move the can of peaches into the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.71738, 0.17355, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.48971, -0.0437, 0.12896), (0.0, 0.0, 0.926179, 0.377084)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.butter = library_object("butter", (0.31577, 0.24586, 0.15541), (0.0, 0.0, 0.958216, 0.286047))
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.50344, 0.19487, 0.15789), (0.0, 0.0, 0.987144, 0.159832)
        )
        self.scene.av_plant_succulent_02 = library_object(
            "av_plant_succulent_02", (0.38961, -0.21533, 0.129), (0.0, 0.0, 0.940728, 0.339163)
        )
        self.scene.bagel_00 = library_object("bagel_00", (0.39746, 0.19592, 0.14458), (0.0, 0.0, 0.601981, 0.798511))
        self.scene.lime01 = library_object("lime01", (0.34205, -0.10808, 0.12922), (0.0, 0.0, 0.397539, 0.917585))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_peaches")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
