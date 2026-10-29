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
class PickPlace034EnvCfg(VLABenchmarkEnvCfg):
    """Remove the spoon from the metal bin, on a console table."""

    instruction = "remove the spoon from the metal bin"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41359, 0.25881, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_b04 = library_object("bin_b04", (0.43709, -0.00792, 0.069), (0.0, 0.0, 0.631171, 0.775643))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.32611, 0.26715, 0.069), (0.0, 0.0, -0.999632, 0.027124)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon = library_object("spoon", (0.43709, -0.00792, 0.09178), (0.0, 0.0, -0.805886, 0.592071))
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.34361, -0.27524, 0.10039), (0.0, 0.0, -0.740057, 0.672544)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.30713, 0.15976, 0.14903), (0.0, 0.0, 0.699895, 0.714245)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("spoon")],
                "container": SceneEntityCfg("bin_b04"),
                "require_gripper_detached": True,
            },
        )
