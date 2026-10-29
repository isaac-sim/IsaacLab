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
class PickPlace222EnvCfg(VLABenchmarkEnvCfg):
    """Put the avocado in the metal bin, on a console table."""

    instruction = "put the avocado in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.41164, 0.08992, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.37111, 0.01489, 0.198), (0.0, 0.0, 0.997246, 0.074168))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.47347, -0.21352, 0.2246), (0.0, 0.0, -0.193738, 0.981053)
        )
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.49671, -0.00866, 0.29424), (0.0, 0.0, -0.51053, 0.85986)
        )
        self.scene.wood_block = library_object(
            "wood_block", (0.33386, -0.20333, 0.30094), (0.0, 0.0, 0.784195, 0.620514)
        )
        self.scene.bagel_00 = library_object("bagel_00", (0.35255, 0.19991, 0.21358), (0.0, 0.0, -0.213536, 0.976935))
        self.scene.avocado01 = library_object("avocado01", (0.50662, 0.16331, 0.19783), (0.0, 0.0, -0.261119, 0.965307))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("avocado01")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
