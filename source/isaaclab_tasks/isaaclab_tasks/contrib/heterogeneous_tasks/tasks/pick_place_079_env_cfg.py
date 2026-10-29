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
class PickPlace079EnvCfg(VLABenchmarkEnvCfg):
    """Place the white bottle into the metal bin, on a table."""

    instruction = "place the white bottle into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_maple = library_object(
            "table_maple", (0.51501, -0.19272, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.39287, -0.11358, 0.009), (0.0, 0.0, -0.21225, 0.977215))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.52058, 0.13768, 0.009), (0.0, 0.0, 0.835898, 0.548886)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.41459, 0.02818, 0.06002), (0.0, 0.0, 0.962757, 0.270369)
        )
        self.scene.spoon_2 = library_object("spoon_2", (0.40089, -0.23338, 0.01439), (0.0, 0.0, 0.918542, 0.395324))
        self.scene.mug_ycb = library_object("mug_ycb", (0.55337, -0.04031, 0.04969), (0.0, 0.0, 0.795128, 0.606441))
        self.scene.soft_scrub = library_object(
            "soft_scrub", (0.35136, 0.15339, 0.13432), (0.0, 0.0, -0.45076, 0.892645)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
