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
class PickPlace250EnvCfg(VLABenchmarkEnvCfg):
    """Place the plastic bottle into the metal bin, on a console table."""

    instruction = "place the plastic bottle into the metal bin"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.4151, 0.23851, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.44373, -0.12516, 0.069), (0.0, 0.0, 0.921069, 0.3894))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_19 = library_object(
            "spatula_19", (0.39167, 0.04943, 0.07268), (0.0, 0.0, 0.991577, 0.129522)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.37575, -0.02131, 0.069), (0.0, 0.0, -0.134595, 0.990901)
        )
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.39645, -0.25601, 0.06874), (0.0, 0.0, -0.050132, 0.998743)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.51285, 0.13751, 0.06908), (0.0, 0.0, -0.836933, 0.547305)
        )
        self.scene.naturalbostonroundbottle_a01 = library_object(
            "naturalbostonroundbottle_a01", (0.28816, -0.16816, 0.069), (0.0, 0.0, -0.654684, 0.755902)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.36481, 0.21846, 0.13907), (0.0, 0.0, 0.200825, 0.979627)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a01")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
