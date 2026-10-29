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
class PickPlace212EnvCfg(VLABenchmarkEnvCfg):
    """Move the yogurt cup into the metal bin, on a table."""

    instruction = "move the yogurt cup into the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.55533, -0.06202, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a02 = library_object("bin_a02", (0.36134, -0.04602, 0.009), (0.0, 0.0, -0.999851, 0.01728))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.53815, 0.04421, 0.009), (0.0, 0.0, -0.424529, 0.905415)
        )
        self.scene.sugar_box = library_object(
            "sugar_box", (0.36295, -0.15756, 0.09701), (0.0, 0.0, -0.717497, 0.696562)
        )
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.47159, -0.20209, 0.03584), (0.0, 0.0, -0.934062, 0.357111)
        )
        self.scene.milkjug_a03 = library_object(
            "milkjug_a03", (0.37676, 0.11323, 0.009), (0.0, 0.0, -0.756339, 0.65418)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_eraser = library_object("av_eraser", (0.51671, -0.1046, 0.01333), (0.0, 0.0, -0.982537, 0.186069))
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("yogurt_cup")],
                "container": SceneEntityCfg("bin_a02"),
                "require_gripper_detached": True,
            },
        )
