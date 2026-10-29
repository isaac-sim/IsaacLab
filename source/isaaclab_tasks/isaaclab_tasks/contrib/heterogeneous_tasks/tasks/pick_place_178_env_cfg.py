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
class PickPlace178EnvCfg(VLABenchmarkEnvCfg):
    """Remove the butter from the black bowl, on an office desk."""

    instruction = "remove the butter from the black bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.45698, -0.01005, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.39468, 0.12717, 0.00696), (0.0, 0.0, -0.206379, 0.978472)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_19 = library_object(
            "spatula_19", (0.40499, -0.1162, 0.01068), (0.0, 0.0, 0.960358, 0.278771)
        )
        self.scene.butter = library_object("butter", (0.39468, 0.12717, 0.04741), (0.0, 0.0, 0.64281, 0.766026))
        self.scene.tuna_can = library_object("tuna_can", (0.56392, -0.05654, 0.02376), (0.0, 0.0, 0.646238, 0.763136))
        self.scene.milk_carton = library_object(
            "milk_carton", (0.54583, 0.06732, 0.10218), (0.0, 0.0, -0.939336, 0.342998)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.30957, -0.20674, 0.04842), (0.0, 0.0, -0.923842, 0.382774)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("butter")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
