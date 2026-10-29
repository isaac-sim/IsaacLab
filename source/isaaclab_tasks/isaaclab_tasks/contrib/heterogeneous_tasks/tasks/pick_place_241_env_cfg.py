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
class PickPlace241EnvCfg(VLABenchmarkEnvCfg):
    """Place the wooden candle holder into the metal bowl, on a table."""

    instruction = "place the wooden candle holder into the metal bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_maple = library_object(
            "table_maple", (0.6753, -0.0315, -0.347), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.37251, -0.03759, 0.00925), (0.0, 0.0, -0.999857, 0.016886)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_09 = library_object(
            "spatula_09", (0.41277, -0.20002, 0.01573), (0.0, 0.0, -0.020539, 0.999789)
        )
        self.scene.fstylejug_a01 = library_object(
            "fstylejug_a01", (0.40606, 0.16702, 0.009), (0.0, 0.0, 0.223403, 0.974726)
        )
        self.scene.scissors = library_object("scissors", (0.5355, 0.11334, 0.01686), (0.0, 0.0, 0.017936, 0.999839))
        self.scene.milk_carton = library_object(
            "milk_carton", (0.55401, -0.06741, 0.10418), (0.0, 0.0, 0.233559, 0.972343)
        )
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.30937, 0.26687, 0.009), (0.0, 0.0, 0.0613, 0.998119)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_woodenchsmall")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
