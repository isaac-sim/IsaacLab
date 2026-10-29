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
class PickPlace017EnvCfg(VLABenchmarkEnvCfg):
    """Put the chocolate pudding mix on the plate, on an open shelf."""

    instruction = "put the chocolate pudding mix on the plate"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_openstorage = library_object(
            "contemporary_openstorage", (0.48729, -0.09306, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.plate_small = library_object(
            "plate_small", (0.40679, -0.00553, -0.03109), (0.0, 0.0, -0.015782, 0.999875)
        )
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.44659, 0.22972, -0.03103), (0.0, 0.0, -0.418641, 0.908152)
        )
        self.scene.fstylejug_a01 = library_object(
            "fstylejug_a01", (0.32602, 0.21646, -0.031), (0.0, 0.0, 0.685007, 0.728537)
        )
        self.scene.canned_tuna = library_object(
            "canned_tuna", (0.41552, -0.25676, -0.01471), (0.0, 0.0, 0.98486, 0.17335)
        )
        self.scene.spoon = library_object("spoon", (0.30659, -0.23341, -0.02222), (0.0, 0.0, -0.594605, 0.804018))
        self.scene.chocolate_pudding_mix = library_object(
            "chocolate_pudding_mix", (0.52018, -0.2255, -0.00626), (0.0, 0.0, -0.719002, 0.695008)
        )
        self.scene.red_block = library_object("red_block", (0.51833, 0.22932, -0.0075), (0.0, 0.0, 0.329367, 0.944202))
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("chocolate_pudding_mix")],
                "reference": SceneEntityCfg("plate_small"),
                "require_gripper_detached": True,
            },
        )
