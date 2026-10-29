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
class PickPlace175EnvCfg(VLABenchmarkEnvCfg):
    """Place the plastic bottle into the white bowl, on a file cabinet."""

    instruction = "place the plastic bottle into the white bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_laterialfile = library_object(
            "contemporary_laterialfile", (0.5366, -0.01462, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.42024, -0.09842, -0.0312), (0.0, 0.0, -0.986315, 0.164874)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a04 = library_object(
            "whitepackerbottle_a04", (0.5616, 0.07928, -0.031), (0.0, 0.0, -0.503671, 0.863895)
        )
        self.scene.mug = library_object("mug", (0.34407, 0.22594, -0.03082), (0.0, 0.0, 0.14902, 0.988834))
        self.scene.av_p_glassware_short = library_object(
            "av_p_glassware_short", (0.36409, 0.11442, -0.031), (0.0, 0.0, -0.503157, 0.864195)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.4602, 0.19392, -0.031), (0.0, 0.0, -0.998149, 0.060815)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
