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
class PickPlace255EnvCfg(VLABenchmarkEnvCfg):
    """Place the plastic bottle upright on top of the book, on a conference table."""

    instruction = "place the plastic bottle upright on top of the book"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.75375, 0.69264, -0.6971), (0.0, 0.0, 0.0, 1.0), kinematic=True)
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_77 = library_object(
            "av_book_77", (0.44804, -0.00678, 0.129), (0.0, 0.0, -0.808633, 0.588314)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cheez_it = library_object("cheez_it", (0.43388, 0.21371, 0.23574), (0.0, 0.0, 0.672841, 0.739787))
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.48463, -0.21905, 0.129), (0.0, 0.0, 0.756919, 0.653509)
        )
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.29221, -0.21461, 0.129), (0.0, 0.0, 0.71195, 0.70223)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.39936, -0.26214, 0.129), (0.0, 0.0, -0.62198, 0.783033)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.30495, 0.15817, 0.20903), (0.0, 0.0, -0.836083, 0.548603)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                "reference": SceneEntityCfg("av_book_77"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
