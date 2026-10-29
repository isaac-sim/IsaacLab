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
class PickPlace221EnvCfg(VLABenchmarkEnvCfg):
    """Pick the white bottle out of the metal bin and put it down, on an end table."""

    instruction = "pick the white bottle out of the metal bin and put it down"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.44224, 0.03768, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.38615, 0.07695, -0.001), (0.0, 0.0, -0.636447, 0.771321))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.38615, 0.07695, 0.013), (0.0, 0.0, 0.72671, 0.686944)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.54816, -0.13984, 0.01576), (0.0, 0.0, -0.869144, 0.49456))
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.53368, 0.18446, -0.001), (0.0, 0.0, -0.953788, 0.300481)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.4445, -0.15605, -0.001), (0.0, 0.0, -0.214192, 0.976792)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.58737, 0.07745, 0.08213), (0.0, 0.0, -0.997351, 0.072738)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a02")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
