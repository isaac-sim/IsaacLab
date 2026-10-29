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
class PickPlace016EnvCfg(VLABenchmarkEnvCfg):
    """Put the chess queen and the barbecue sauce bottle in the red bowl, on an office desk."""

    instruction = "put the chess queen and the barbecue sauce bottle in the red bowl"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowhalfdesk = library_object(
            "contemporary_lowhalfdesk", (0.46937, 0.11236, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.34284, -0.03269, 0.03448), (0.0, 0.0, -0.72479, 0.68897))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_03 = library_object(
            "spatula_03", (0.44927, -0.17767, 0.01373), (0.0, 0.0, -0.943391, 0.331682)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.46801, -0.00274, 0.007), (0.0, 0.0, -1.0, 0.000553)
        )
        self.scene.pumpkinlarge = library_object(
            "pumpkinlarge", (0.50176, 0.19843, 0.00676), (0.0, 0.0, -0.312783, 0.949825)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.34868, 0.16226, 0.00708), (0.0, 0.0, -0.867665, 0.49715)
        )
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.45287, 0.09992, 0.08116), (0.0, 0.0, 0.146688, 0.989183)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_queen"), SceneEntityCfg("bbq_sauce_bottle")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
