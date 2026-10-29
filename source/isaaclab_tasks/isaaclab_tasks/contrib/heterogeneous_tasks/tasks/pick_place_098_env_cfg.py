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
class PickPlace098EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the raisin box and put it in the red bowl, on an open shelf."""

    instruction = "pick up the raisin box and put it in the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.49645, 0.01913, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.45642, 0.11888, -0.00352), (0.0, 0.0, -0.999981, 0.006227))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.spam_can = library_object("spam_can", (0.52694, -0.03185, 0.01082), (0.0, 0.0, 0.984072, 0.177773))
        self.scene.raisin_box = library_object(
            "raisin_box", (0.52612, -0.18846, 0.03059), (0.0, 0.0, -0.626302, 0.77958)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.28736, 0.19215, -0.03092), (0.0, 0.0, -0.127498, 0.991839)
        )
        self.scene.measuring_spoon = library_object(
            "measuring_spoon", (0.38267, -0.1468, -0.01666), (0.0, 0.0, 0.747125, 0.664683)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("raisin_box")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
