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
class PickPlace168EnvCfg(VLABenchmarkEnvCfg):
    """Put the can of tuna in the metal bin, on a console table."""

    instruction = "put the can of tuna in the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.40611, 0.26655, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.38024, 0.13968, 0.198), (0.0, 0.0, 0.793052, 0.609154))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_03 = library_object(
            "spatula_03", (0.30965, -0.14097, 0.20473), (0.0, 0.0, -0.777502, 0.62888)
        )
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.50371, 0.15725, 0.22484), (0.0, 0.0, -0.557678, 0.830057)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.4549, -0.00909, 0.21429), (0.0, 0.0, -0.414506, 0.910047)
        )
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.44899, -0.20287, 0.22403), (0.0, 0.0, 0.766151, 0.64266)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_tuna")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
