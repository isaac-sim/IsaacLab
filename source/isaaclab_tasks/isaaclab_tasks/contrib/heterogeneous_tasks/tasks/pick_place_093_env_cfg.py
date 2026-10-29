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
class PickPlace093EnvCfg(VLABenchmarkEnvCfg):
    """First put the wooden box in the red bowl, then put the chocolate pudding mix in it, on a conference table."""

    instruction = "first put the wooden box in the red bowl, then put the chocolate pudding mix in it"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.72759, -0.59797, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.bowl = library_object("bowl", (0.41687, 0.15064, 0.15648), (0.0, 0.0, 0.681609, 0.731717))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.33179, -0.07969, 0.129), (0.0, 0.0, 0.423097, 0.906085)
        )
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.55464, -0.15044, 0.12897), (0.0, 0.0, 0.126002, 0.99203)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.granola_bars = library_object(
            "granola_bars", (0.45064, -0.20941, 0.191), (0.0, 0.0, 0.534363, 0.845255)
        )
        self.scene.storage_box = library_object(
            "storage_box", (0.46047, -0.09457, 0.12887), (0.0, 0.0, -0.889931, 0.456095)
        )
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.55547, 0.03569, 0.15374), (0.0, 0.0, 0.897657, 0.440694)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("storage_box"), SceneEntityCfg("chocolate_pudding_mix")],
                "container": SceneEntityCfg("bowl"),
            },
        )
