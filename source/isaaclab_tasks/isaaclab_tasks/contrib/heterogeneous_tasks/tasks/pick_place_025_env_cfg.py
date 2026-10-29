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
class PickPlace025EnvCfg(VLABenchmarkEnvCfg):
    """Put the kiwi in the metal bin, on a table."""

    instruction = "put the kiwi in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_maple = library_object(
            "table_maple", (0.545, 0.10895, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.4679, 0.19601, 0.009), (0.0, 0.0, -0.081679, 0.996659))
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.39471, -0.05925, 0.009), (0.0, 0.0, -0.847193, 0.531285)
        )
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.34207, 0.05125, 0.00899), (0.0, 0.0, -0.67297, 0.739669))
        self.scene.ranch_dressing = library_object(
            "ranch_dressing", (0.31793, 0.18889, 0.08272), (0.0, 0.0, 0.111773, 0.993734)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.52073, -0.2367, 0.009), (0.0, 0.0, 0.882841, 0.469671)
        )
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.37481, -0.22181, 0.03779), (0.0, 0.0, 0.396004, 0.918249)
        )
        self.scene.potato_masher = library_object(
            "potato_masher", (0.4856, -0.12782, 0.00942), (0.0, 0.0, 0.966074, 0.258265)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_kiwi01")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
