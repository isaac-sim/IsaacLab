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
class PickPlace232EnvCfg(VLABenchmarkEnvCfg):
    """Remove the can of peaches from the white bowl, on an office desk."""

    instruction = "remove the can of peaches from the white bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.43657, -0.08132, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.41711, 0.01751, 0.0068), (0.0, 0.0, -0.984574, 0.174971)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cheez_it = library_object("cheez_it", (0.33291, -0.20108, 0.11374), (0.0, 0.0, -0.697977, 0.71612))
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.41724, 0.01756, 0.04989), (0.0, 0.0, -0.024607, 0.999697)
        )
        self.scene.milk_carton = library_object(
            "milk_carton", (0.31322, 0.16127, 0.10218), (0.0, 0.0, 0.175619, 0.984458)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard = library_object("mustard", (0.52398, -0.20474, 0.1027), (0.0, 0.0, 0.706509, 0.707704))
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("canned_peaches")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
