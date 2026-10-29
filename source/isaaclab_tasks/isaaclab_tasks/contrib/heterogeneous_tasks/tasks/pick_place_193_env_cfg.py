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
class PickPlace193EnvCfg(VLABenchmarkEnvCfg):
    """Place the ketchup bottle upright on top of the coaster, on a dining table."""

    instruction = "place the ketchup bottle upright on top of the coaster"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eastrural_table = library_object(
            "av_eastrural_table", (0.62798, -0.44138, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.34132, -0.09159, 0.09848), (0.0, 0.0, 0.997846, 0.065603))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.5033, -0.05172, 0.071), (0.0, 0.0, 0.71867, 0.695352)
        )
        self.scene.spatula_09 = library_object(
            "spatula_09", (0.43288, 0.08681, 0.07773), (0.0, 0.0, -0.037048, 0.999313)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.43252, 0.19126, 0.1453), (0.0, 0.0, 0.288591, 0.957452)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.50806, -0.17902, 0.11242), (0.0, 0.0, 0.613064, 0.790033)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("ketchup_bottle")],
                "reference": SceneEntityCfg("av_coaster_hexagon"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
