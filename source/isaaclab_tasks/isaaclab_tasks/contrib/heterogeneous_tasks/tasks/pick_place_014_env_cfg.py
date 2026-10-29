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
class PickPlace014EnvCfg(VLABenchmarkEnvCfg):
    """Place the granola bar box on top of the clay plate, on a storage cabinet."""

    instruction = "place the granola bar box on top of the clay plate"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.42484, 0.0893, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.clay_plates = library_object(
            "clay_plates", (0.41865, 0.00034, -0.03117), (0.0, 0.0, -0.998922, 0.046426)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object(
            "orange_02", (0.40891, -0.24841, -0.03094), (0.0, 0.0, -0.982781, 0.184774)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.37777, 0.25831, 0.031), (0.0, 0.0, -0.639432, 0.768847)
        )
        self.scene.storage_box = library_object(
            "storage_box", (0.28994, -0.25823, -0.03113), (0.0, 0.0, 0.682924, 0.730489)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.52347, -0.23494, -0.00626), (0.0, 0.0, 0.999138, 0.041502)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("granola_bars")],
                "reference": SceneEntityCfg("clay_plates"),
                "require_gripper_detached": True,
            },
        )
