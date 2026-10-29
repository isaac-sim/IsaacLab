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
class PickPlace214EnvCfg(VLABenchmarkEnvCfg):
    """Put the wooden candle holder in the decorative bowl, then the raisin box, on a table."""

    instruction = "put the wooden candle holder in the decorative bowl, then the raisin box"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_maple = library_object(
            "table_maple", (0.66603, -0.01217, -0.347), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.42369, 0.00439, 0.00904), (0.0, 0.0, -0.01696, 0.999856)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.raisin_box = library_object(
            "raisin_box", (0.3325, 0.24395, 0.07059), (0.0, 0.0, -0.168086, 0.985772)
        )
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.44351, 0.24003, 0.009), (0.0, 0.0, -0.998003, 0.063163)
        )
        self.scene.av_woodenchmedium = library_object(
            "av_woodenchmedium", (0.32975, -0.19752, 0.00901), (0.0, 0.0, -0.639147, 0.769085)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("av_woodenchmedium"), SceneEntityCfg("raisin_box")],
                "container": SceneEntityCfg("av_decorativebowl"),
            },
        )
