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
class PickPlace065EnvCfg(VLABenchmarkEnvCfg):
    """Put the granola bar box in the wooden bowl, on a table."""

    instruction = "put the granola bar box in the wooden bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.53747, 0.19487, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.wooden_bowl = library_object(
            "wooden_bowl", (0.40529, 0.10515, 0.00892), (0.0, 0.0, 0.021709, 0.999764)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.granola_bars = library_object(
            "granola_bars", (0.36753, -0.16018, 0.071), (0.0, 0.0, 0.984417, 0.175851)
        )
        self.scene.spoon_2 = library_object("spoon_2", (0.54321, -0.11739, 0.01439), (0.0, 0.0, -0.461353, 0.887217))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.avocado01 = library_object(
            "avocado01", (0.47971, -0.24589, 0.00883), (0.0, 0.0, -0.592679, 0.805439)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("granola_bars")],
                "container": SceneEntityCfg("wooden_bowl"),
                "require_gripper_detached": True,
            },
        )
