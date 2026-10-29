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
class PickPlace216EnvCfg(VLABenchmarkEnvCfg):
    """Take the chess bishop out of the metal bin, on an office desk."""

    instruction = "take the chess bishop out of the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.47397, 0.2536, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.477, 0.09922, 0.007), (0.0, 0.0, -0.611841, 0.790981))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.35847, -0.10345, 0.08072), (0.0, 0.0, -0.643084, 0.765795)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.477, 0.09922, 0.021), (0.0, 0.0, -0.349438, 0.93696)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.49132, -0.16743, 0.03579), (0.0, 0.0, 0.183072, 0.9831)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.39278, -0.20515, 0.03611), (0.0, 0.0, -0.064814, 0.997897)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("av_chess_bishop")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
