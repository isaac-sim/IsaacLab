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
class PickPlace046EnvCfg(VLABenchmarkEnvCfg):
    """Put the avocado in the metal bowl, on a desk."""

    instruction = "put the avocado in the metal bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.49508, -0.4321, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.38737, 0.0481, 0.09025), (0.0, 0.0, -0.947487, 0.319793)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_12 = library_object(
            "spatula_12", (0.53886, 0.1049, 0.09367), (0.0, 0.0, -0.619882, 0.784695)
        )
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.40579, -0.18669, 0.1166), (0.0, 0.0, 0.696952, 0.717118)
        )
        self.scene.soup_can = library_object("soup_can", (0.34515, 0.18268, 0.09006), (0.0, 0.0, 0.284119, 0.958789))
        self.scene.ketchup_bottle = library_object(
            "ketchup_bottle", (0.52035, -0.14553, 0.1643), (0.0, 0.0, 0.739073, 0.673625)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.28686, -0.26037, 0.12139), (0.0, 0.0, -0.826627, 0.56275)
        )
        self.scene.avocado01 = library_object("avocado01", (0.4571, -0.0699, 0.08983), (0.0, 0.0, 0.237954, 0.971277))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("avocado01")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
