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
class PickPlace005EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the lime and put it in the pan, on a desk."""

    instruction = "pick up the lime and put it in the pan"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.desk_01 = library_object(
            "desk_01", (0.50524, 0.60417, -0.69646), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.av_pan_d = library_object("av_pan_d", (0.42515, -0.05703, 0.09026), (0.0, 0.0, 0.99929, 0.037675))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.27894, 0.25622, 0.08997), (0.0, 0.0, -0.035785, 0.99936)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.34612, -0.24289, 0.09944), (0.0, 0.0, 0.573546, 0.819173)
        )
        self.scene.lime01 = library_object("lime01", (0.48219, -0.25128, 0.09022), (0.0, 0.0, 0.488966, 0.872303))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("lime01")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
