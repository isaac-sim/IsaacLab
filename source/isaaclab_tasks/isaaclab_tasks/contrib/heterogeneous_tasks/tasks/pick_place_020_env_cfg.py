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
class PickPlace020EnvCfg(VLABenchmarkEnvCfg):
    """Move the spoon into the red bowl, on a table."""

    instruction = "move the spoon into the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.51563, 0.06531, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.46316, -0.12458, 0.03648), (0.0, 0.0, -0.949502, 0.313761))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_11 = library_object("spatula_11", (0.4595, 0.12662, 0.01573), (0.0, 0.0, 0.962194, 0.272364))
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.54213, 0.08461, 0.08272), (0.0, 0.0, 0.925838, 0.377922)
        )
        self.scene.tuna_can = library_object("tuna_can", (0.32422, -0.20776, 0.02576), (0.0, 0.0, -0.249528, 0.968368))
        self.scene.spoon_1 = library_object("spoon_1", (0.36085, 0.25489, 0.01966), (0.0, 0.0, -0.023162, 0.999732))
        self.scene.av_eraser = library_object("av_eraser", (0.33636, 0.05297, 0.01333), (0.0, 0.0, 0.579019, 0.815314))
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.28663, 0.14939, 0.08316), (0.0, 0.0, 0.695096, 0.718917)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_1")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
