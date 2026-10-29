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
class PickPlace228EnvCfg(VLABenchmarkEnvCfg):
    """Pick the foam roller out of the white bowl and put it down, on a conference table."""

    instruction = "pick the foam roller out of the white bowl and put it down"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.69787, -0.17885, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.42612, -0.08181, 0.1288), (0.0, 0.0, 0.71521, 0.698909)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_09 = library_object(
            "spatula_09", (0.46324, 0.17198, 0.13573), (0.0, 0.0, 0.983986, 0.178247)
        )
        self.scene.orange_02 = library_object("orange_02", (0.35338, 0.1013, 0.12906), (0.0, 0.0, 0.997166, 0.075239))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.59244, -0.01626, 0.2033), (0.0, 0.0, 0.998953, 0.045745)
        )
        self.scene.yellow_block = library_object(
            "yellow_block", (0.44595, -0.23987, 0.1525), (0.0, 0.0, 0.995141, 0.098463)
        )
        self.scene.foam_roller = library_object(
            "foam_roller", (0.42612, -0.08195, 0.14305), (0.0, 0.0, 0.744032, 0.668144)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("foam_roller")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
