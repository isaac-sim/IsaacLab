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
class PickPlace233EnvCfg(VLABenchmarkEnvCfg):
    """Move the can of pineapple into the red bowl, on an open shelf."""

    instruction = "move the can of pineapple into the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.43631, -0.0095, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.36062, 0.1521, -0.00352), (0.0, 0.0, -0.512054, 0.858953))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.frozen_waffles = library_object(
            "frozen_waffles", (0.52319, -0.07291, -0.03102), (0.0, 0.0, 0.237588, 0.971366)
        )
        self.scene.whitepackerbottle_a01 = library_object(
            "whitepackerbottle_a01", (0.42598, -0.26299, -0.031), (0.0, 0.0, 0.544318, 0.838879)
        )
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.54985, 0.07844, -0.00219), (0.0, 0.0, -0.777162, 0.629301)
        )
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.48649, 0.2412, -0.031), (0.0, 0.0, -0.828902, 0.559394)
        )
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.36417, -0.17176, -0.03104), (0.0, 0.0, 0.049975, 0.99875)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("pineapple_slices_can")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
