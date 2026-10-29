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
class PickPlace088EnvCfg(VLABenchmarkEnvCfg):
    """Move the white bottle into the salt bowl, on a console table."""

    instruction = "move the white bottle into the salt bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41715, 0.177, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.av_salt_bowl = library_object(
            "av_salt_bowl", (0.45979, 0.05525, 0.06906), (0.0, 0.0, -0.962678, 0.270649)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.34551, -0.23279, 0.069), (0.0, 0.0, 0.221394, 0.975185)
        )
        self.scene.av_p_glassware_tall = library_object(
            "av_p_glassware_tall", (0.31099, 0.21317, 0.069), (0.0, 0.0, 0.235614, 0.971847)
        )
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.35539, -0.08941, 0.1433), (0.0, 0.0, 0.805833, 0.592144)
        )
        self.scene.mug_ycb = library_object("mug_ycb", (0.46872, -0.17078, 0.10969), (0.0, 0.0, -0.344735, 0.9387))
        self.scene.brick = library_object("brick", (0.44497, 0.20976, 0.09457), (0.0, 0.0, 0.85384, 0.520536))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("whitepackerbottle_a04")],
                "container": SceneEntityCfg("av_salt_bowl"),
                "require_gripper_detached": True,
            },
        )
