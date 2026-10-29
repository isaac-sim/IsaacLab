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
class PickPlace096EnvCfg(VLABenchmarkEnvCfg):
    """First put the spoon in the decorative bowl, then put the mustard bottle in it, on an office desk."""

    instruction = "first put the spoon in the decorative bowl, then put the mustard bottle in it"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.42796, -0.31079, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_decorativebowl = library_object(
            "av_decorativebowl", (0.42903, 0.09719, 0.00704), (0.0, 0.0, -0.988946, 0.148277)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.54536, -0.11851, 0.007), (0.0, 0.0, 0.648004, 0.761637)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon = library_object("spoon", (0.36489, -0.15764, 0.01578), (0.0, 0.0, 0.98978, 0.142601))
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.43125, -0.24125, 0.08703), (0.0, 0.0, -0.487637, 0.873046)
        )
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("spoon"), SceneEntityCfg("mustard_bottle")],
                "container": SceneEntityCfg("av_decorativebowl"),
            },
        )
