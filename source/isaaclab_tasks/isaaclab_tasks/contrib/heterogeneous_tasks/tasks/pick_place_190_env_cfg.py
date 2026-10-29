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
class PickPlace190EnvCfg(VLABenchmarkEnvCfg):
    """Move the milk jug into the metal bin, on a sofa table."""

    instruction = "move the milk jug into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_sofatable = library_object(
            "appleseed_sofatable", (0.4403, 0.30669, -0.69682), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.43907, -0.11024, 0.058), (0.0, 0.0, -0.472013, 0.881592))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.parmesan_cheese_canister = library_object(
            "parmesan_cheese_canister", (0.30915, -0.01193, 0.10943), (0.0, 0.0, -0.365359, 0.930867)
        )
        self.scene.milkjug_a01 = library_object("milkjug_a01", (0.482, 0.02732, 0.058), (0.0, 0.0, 0.189162, 0.981946))
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.56675, 0.09057, 0.08726), (0.0, 0.0, 0.13798, 0.990435)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.37294, 0.15911, 0.14113), (0.0, 0.0, 0.803097, 0.595848)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("milkjug_a01")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
