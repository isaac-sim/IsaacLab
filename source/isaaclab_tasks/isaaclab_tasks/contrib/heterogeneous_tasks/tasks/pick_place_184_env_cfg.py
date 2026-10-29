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
class PickPlace184EnvCfg(VLABenchmarkEnvCfg):
    """Move the white bottle into the black bowl, on a conference table."""

    instruction = "move the white bottle into the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.steam = library_object("steam", (2.33592, 0.09385, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True)
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.42849, 0.07219, 0.10894), (0.0, 0.0, -0.634674, 0.77278)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.53542, -0.23432, 0.109), (0.0, 0.0, -0.117145, 0.993115)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.5396, -0.1247, 0.12529), (0.0, 0.0, -0.467228, 0.884137)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.bagel_00 = library_object("bagel_00", (0.3661, -0.20066, 0.12458), (0.0, 0.0, -0.980917, 0.194426))
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.29622, 0.24394, 0.10906), (0.0, 0.0, -0.986595, 0.163186)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a03")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "require_gripper_detached": True,
            },
        )
