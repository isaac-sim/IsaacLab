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
class PickPlace148EnvCfg(VLABenchmarkEnvCfg):
    """Place the orange can into the metal bin, on a table."""

    instruction = "place the orange can into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_black = library_object(
            "table_black", (0.51564, -0.18019, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.49401, 0.05482, 0.009), (0.0, 0.0, 0.837733, 0.54608))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.glasses = library_object("glasses", (0.40512, 0.20443, 0.03947), (0.0, 0.0, 0.747495, 0.664267))
        self.scene.soup_can = library_object("soup_can", (0.35786, -0.25036, 0.00906), (0.0, 0.0, -0.989184, 0.146682))
        self.scene.spoon_2 = library_object("spoon_2", (0.45467, -0.24567, 0.01439), (0.0, 0.0, 0.930277, 0.366858))
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.33116, -0.05125, 0.009), (0.0, 0.0, -0.603121, 0.797649)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.49784, -0.14852, 0.03811), (0.0, 0.0, -0.958076, 0.286515)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("soup_can")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
