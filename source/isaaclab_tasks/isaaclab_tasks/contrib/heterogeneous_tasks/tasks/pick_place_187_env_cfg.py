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
class PickPlace187EnvCfg(VLABenchmarkEnvCfg):
    """Place the kiwi on the popcorn box, on a table."""

    instruction = "place the kiwi on the popcorn box"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_willow = library_object(
            "av_willow", (0.58589, 0.38231, -0.69705), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.33843, -0.18341, 0.069), (0.0, 0.0, 0.203284, 0.97912))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.48186, 0.05673, 0.069), (0.0, 0.0, -0.149907, 0.9887)
        )
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.48071, 0.21297, 0.06899), (0.0, 0.0, 0.875477, 0.48326))
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.29782, -0.05583, 0.12043), (0.0, 0.0, -0.284374, 0.958714)
        )
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.42603, -0.07405, 0.11149), (0.0, 0.0, 0.556306, 0.830977)
        )
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.34359, 0.16155, 0.09503), (0.0, 0.0, 0.577258, 0.816562)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.55438, -0.11701, 0.069), (0.0, 0.0, 0.359592, 0.93311)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "reference": SceneEntityCfg("popcorn_box"),
                "require_gripper_detached": True,
            },
        )
