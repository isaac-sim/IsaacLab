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
class PickPlace231EnvCfg(VLABenchmarkEnvCfg):
    """Put the mustard bottle in the black bowl, then the pumpkin, on an office desk."""

    instruction = "put the mustard bottle in the black bowl, then the pumpkin"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.47358, 0.55221, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.42828, -0.00531, 0.00694), (0.0, 0.0, 0.082428, 0.996597)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.51631, -0.21891, 0.00674), (0.0, 0.0, -0.320547, 0.947233)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard = library_object("mustard", (0.30899, 0.22831, 0.1027), (0.0, 0.0, -0.766328, 0.64245))
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("mustard"), SceneEntityCfg("pumpkinsmall")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
            },
        )
