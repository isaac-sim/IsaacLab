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
class PickPlace141EnvCfg(VLABenchmarkEnvCfg):
    """Place both the orange can and the red block into the metal bin, on a console table."""

    instruction = "place both the orange can and the red block into the metal bin"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.4104, 0.00752, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.40289, 0.01023, 0.198), (0.0, 0.0, -0.743459, 0.668781))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.soup_can = library_object("soup_can", (0.30601, -0.24931, 0.19806), (0.0, 0.0, -0.357495, 0.933915))
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.39011, 0.23115, 0.20744), (0.0, 0.0, 0.768349, 0.640031)
        )
        self.scene.red_block = library_object("red_block", (0.50274, -0.25504, 0.2215), (0.0, 0.0, -0.269452, 0.963014))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("soup_can"), SceneEntityCfg("red_block")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
