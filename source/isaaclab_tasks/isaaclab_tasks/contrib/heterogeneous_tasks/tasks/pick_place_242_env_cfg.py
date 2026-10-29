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
class PickPlace242EnvCfg(VLABenchmarkEnvCfg):
    """Place both the orange and the mayonnaise bottle into the red bowl, on a table."""

    instruction = "place both the orange and the mayonnaise bottle into the red bowl"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.54507, 0.05578, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.47738, 0.00706, 0.03648), (0.0, 0.0, -0.880399, 0.474234))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.sugar_box = library_object("sugar_box", (0.43493, 0.20249, 0.09701), (0.0, 0.0, 0.457724, 0.889095))
        self.scene.orange_02 = library_object("orange_02", (0.2967, -0.1287, 0.00906), (0.0, 0.0, 0.188765, 0.982022))
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.35819, 0.13368, 0.009), (0.0, 0.0, -0.137535, 0.990497)
        )
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.46471, -0.23531, 0.02566), (0.0, 0.0, -0.499757, 0.866165)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.3923, -0.14356, 0.08295), (0.0, 0.0, 0.647629, 0.761956)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("orange_02"), SceneEntityCfg("mayonnaise_bottle")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
