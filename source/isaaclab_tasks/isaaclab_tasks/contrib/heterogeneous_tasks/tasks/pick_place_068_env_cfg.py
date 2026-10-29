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
class PickPlace068EnvCfg(VLABenchmarkEnvCfg):
    """Put the can of tuna in the red bowl, on a storage cabinet."""

    instruction = "put the can of tuna in the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lockedstorage = library_object(
            "contemporary_lockedstorage", (0.4863, 0.00661, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.39243, 0.02976, -0.00352), (0.0, 0.0, 0.768903, 0.639366))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.butter = library_object("butter", (0.47073, 0.1687, -0.00459), (0.0, 0.0, -0.824652, 0.56564))
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.53296, -0.1401, -0.01471), (0.0, 0.0, 0.984659, 0.174489)
        )
        self.scene.av_chess_king = library_object(
            "av_chess_king", (0.45839, -0.22816, -0.031), (0.0, 0.0, -0.269041, 0.963129)
        )
        self.scene.mayonnaise_bottle = library_object(
            "mayonnaise_bottle", (0.30186, -0.2599, 0.04295), (0.0, 0.0, 0.787813, 0.615915)
        )
        self.scene.potato_masher = library_object(
            "potato_masher", (0.31028, 0.18418, -0.03058), (0.0, 0.0, 0.995115, 0.098724)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("canned_tuna")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
