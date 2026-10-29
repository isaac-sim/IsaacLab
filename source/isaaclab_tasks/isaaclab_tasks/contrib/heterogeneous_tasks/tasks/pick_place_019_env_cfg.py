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
class PickPlace019EnvCfg(VLABenchmarkEnvCfg):
    """Place the measuring spoons into the metal bin, on a console table."""

    instruction = "place the measuring spoons into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41179, 0.35924, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.4278, -0.08273, 0.069), (0.0, 0.0, 0.78851, 0.615022))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.spam_can = library_object("spam_can", (0.50674, 0.21646, 0.11082), (0.0, 0.0, -0.113715, 0.993513))
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.29676, -0.21389, 0.12002), (0.0, 0.0, -0.985335, 0.170629)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.34622, 0.19811, 0.08334), (0.0, 0.0, -0.301303, 0.953528)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("measuring_spoon")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
