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
class PickPlace151EnvCfg(VLABenchmarkEnvCfg):
    """Stack the Rubik's cube on the sugar box, on an office desk."""

    instruction = "stack the Rubik's cube on the sugar box"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.504, 0.0634, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.53557, 0.07638, 0.09501), (0.0, 0.0, -0.989717, 0.143036))
        self.scene.cubebox_a02 = library_object(
            "cubebox_a02", (0.34169, 0.11799, 0.007), (0.0, 0.0, 0.998539, 0.054027)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.30449, -0.05502, 0.03589), (0.0, 0.0, 0.946911, 0.321497)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.36026, -0.18589, 0.00708), (0.0, 0.0, 0.827939, 0.560818)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.49208, -0.12128, 0.03839), (0.0, 0.0, 0.677713, 0.735327)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("rubiks_cube")],
                "reference": SceneEntityCfg("sugar_box"),
                "require_gripper_detached": True,
            },
        )
