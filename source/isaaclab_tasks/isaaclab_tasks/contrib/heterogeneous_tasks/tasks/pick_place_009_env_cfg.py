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
class PickPlace009EnvCfg(VLABenchmarkEnvCfg):
    """Place both the Rubik's cube and the barbecue sauce bottle into the metal bowl, on an office desk."""

    instruction = "place both the Rubik's cube and the barbecue sauce bottle into the metal bowl"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.43713, -0.04354, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.43025, 0.16591, 0.00725), (0.0, 0.0, -0.658959, 0.752179)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.fstylejug_a02 = library_object(
            "fstylejug_a02", (0.42317, -0.20978, 0.007), (0.0, 0.0, 0.367219, 0.930135)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.56363, -0.0546, 0.02329), (0.0, 0.0, -0.121142, 0.992635)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.30373, -0.01238, 0.00708), (0.0, 0.0, 0.802621, 0.59649)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.38328, -0.01465, 0.03839), (0.0, 0.0, -0.69359, 0.72037)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.46794, -0.09534, 0.00706), (0.0, 0.0, -0.773361, 0.633966)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("rubiks_cube"), SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
