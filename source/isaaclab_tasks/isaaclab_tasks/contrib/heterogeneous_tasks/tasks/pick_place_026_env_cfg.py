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
class PickPlace026EnvCfg(VLABenchmarkEnvCfg):
    """Put the lime in the salt bowl, then the can of green beans, on a table."""

    instruction = "put the lime in the salt bowl, then the can of green beans"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.52618, 0.12911, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.40583, 0.06706, 0.00906), (0.0, 0.0, -0.351131, 0.936326)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.38587, -0.26012, 0.03584), (0.0, 0.0, -0.647375, 0.762172)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.45647, -0.19782, 0.071), (0.0, 0.0, 0.529816, 0.848113)
        )
        self.scene.wood_block = library_object(
            "wood_block", (0.48495, 0.21366, 0.11194), (0.0, 0.0, -0.876112, 0.482108)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.49922, -0.12964, 0.03779), (0.0, 0.0, -0.925505, 0.378735)
        )
        self.scene.lime01 = library_object("lime01", (0.33595, -0.11467, 0.00922), (0.0, 0.0, 0.542318, 0.840173))
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("lime01"), SceneEntityCfg("green_beans_can")],
                "container": SceneEntityCfg("av_salt_bowl"),
            },
        )
