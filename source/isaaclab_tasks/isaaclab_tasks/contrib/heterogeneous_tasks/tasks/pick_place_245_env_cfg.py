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
class PickPlace245EnvCfg(VLABenchmarkEnvCfg):
    """Put the kiwi on the book, on a console table."""

    instruction = "put the kiwi on the book"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_aftonconsole = library_object(
            "av_aftonconsole", (0.42518, -0.30968, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_66 = library_object("av_book_66", (0.43005, -0.12739, 0.109), (0.0, 0.0, 0.968345, 0.249617))
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.50795, 0.15395, 0.10899), (0.0, 0.0, 0.465412, 0.885094))
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.51704, 0.06415, 0.18272), (0.0, 0.0, -0.941585, 0.336776)
        )
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.32497, 0.00536, 0.13781), (0.0, 0.0, -0.837376, 0.546627)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.empty_7 = HeterogeneousTaskManager.placeholder(7)
        self.scene.av_tincan = library_object("av_tincan", (0.33127, 0.17202, 0.109), (0.0, 0.0, 0.448122, 0.893972))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "reference": SceneEntityCfg("av_book_66"),
                "require_gripper_detached": True,
            },
        )
