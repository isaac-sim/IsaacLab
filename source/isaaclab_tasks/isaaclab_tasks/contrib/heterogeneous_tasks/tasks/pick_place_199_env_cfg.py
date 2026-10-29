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
class PickPlace199EnvCfg(VLABenchmarkEnvCfg):
    """Place the plastic bottle upright on top of the book, on an office desk."""

    instruction = "place the plastic bottle upright on top of the book"
    episode_length_s = 75.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.42797, 0.40298, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.av_book_51 = library_object("av_book_51", (0.43942, -0.07643, 0.007), (0.0, 0.0, 0.995748, 0.09212))
        self.scene.sugar_box = library_object("sugar_box", (0.42204, 0.199, 0.09501), (0.0, 0.0, 0.069159, 0.997606))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.milkjug_a02 = library_object(
            "milkjug_a02", (0.31551, 0.20533, 0.007), (0.0, 0.0, -0.999893, 0.014634)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.55719, 0.17447, 0.007), (0.0, 0.0, 0.815755, 0.578397)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                "reference": SceneEntityCfg("av_book_51"),
                "require_gripper_detached": True,
                "require_upright": True,
            },
        )
