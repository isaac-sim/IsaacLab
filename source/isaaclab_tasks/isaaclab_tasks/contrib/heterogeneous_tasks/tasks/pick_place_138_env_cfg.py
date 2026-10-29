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
class PickPlace138EnvCfg(VLABenchmarkEnvCfg):
    """Place the can of peaches into the red bowl, on a dining table."""

    instruction = "place the can of peaches into the red bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.64623, 0.59447, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.37239, 0.16088, 0.10448), (0.0, 0.0, 0.983737, 0.179613))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_13 = library_object(
            "spatula_13", (0.42786, -0.13385, 0.08067), (0.0, 0.0, 0.249506, 0.968373)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.4835, -0.00794, 0.10589), (0.0, 0.0, 0.352947, 0.935643)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.58399, -0.0338, 0.07708), (0.0, 0.0, -0.715405, 0.69871)
        )
        self.scene.bagel_00 = library_object("bagel_00", (0.32483, -0.21033, 0.09258), (0.0, 0.0, 0.064983, 0.997886))
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.56727, 0.09381, 0.16013), (0.0, 0.0, 0.161168, 0.986927)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_peaches")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
