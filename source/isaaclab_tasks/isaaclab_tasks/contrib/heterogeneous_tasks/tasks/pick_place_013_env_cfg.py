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
class PickPlace013EnvCfg(VLABenchmarkEnvCfg):
    """Stack the green block on the wooden box, on a file cabinet."""

    instruction = "stack the green block on the wooden box"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.46663, 0.03152, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.38368, 0.10435, -0.03104), (0.0, 0.0, 0.639952, 0.768415)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.53839, 0.0491, -0.0075), (0.0, 0.0, 0.316999, 0.948426)
        )
        self.scene.raisin_box = library_object("raisin_box", (0.292, -0.25207, 0.03059), (0.0, 0.0, 0.241451, 0.970413))
        self.scene.storage_box = library_object(
            "storage_box", (0.50576, -0.19995, -0.03113), (0.0, 0.0, -0.743873, 0.668321)
        )
        self.scene.naturalbostonroundbottle_a03 = library_object(
            "naturalbostonroundbottle_a03", (0.36989, -0.13409, -0.031), (0.0, 0.0, -0.195709, 0.980662)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.49519, -0.1053, -0.00189), (0.0, 0.0, 0.782678, 0.622427)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("green_block")],
                "reference": SceneEntityCfg("storage_box"),
                "require_gripper_detached": True,
            },
        )
