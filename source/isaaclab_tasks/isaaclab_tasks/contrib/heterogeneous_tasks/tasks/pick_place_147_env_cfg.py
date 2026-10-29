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
class PickPlace147EnvCfg(VLABenchmarkEnvCfg):
    """Put the mustard bottle in the metal bin, on a conference table."""

    instruction = "put the mustard bottle in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.71731, -0.68065, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.bin_a02 = library_object("bin_a02", (0.49565, 0.04339, 0.129), (0.0, 0.0, -0.20314, 0.97915))
        self.scene.av_book_33 = library_object("av_book_33", (0.44371, -0.15045, 0.129), (0.0, 0.0, 0.574966, 0.818177))
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_stawberry = library_object(
            "av_stawberry", (0.27594, -0.05888, 0.12881), (0.0, 0.0, 0.33254, 0.943089)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.storage_box = library_object(
            "storage_box", (0.39622, 0.2088, 0.12887), (0.0, 0.0, 0.916299, 0.400496)
        )
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.3094, 0.09166, 0.129), (0.0, 0.0, -0.242154, 0.970238)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.53487, 0.23497, 0.20903), (0.0, 0.0, 0.844842, 0.535016)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("mustard_bottle")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
