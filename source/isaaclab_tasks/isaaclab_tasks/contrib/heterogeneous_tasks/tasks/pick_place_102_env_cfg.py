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
class PickPlace102EnvCfg(VLABenchmarkEnvCfg):
    """Put the plastic bottle in the pan, on a conference table."""

    instruction = "put the plastic bottle in the pan"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.68734, -0.60028, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.av_pan_d = library_object("av_pan_d", (0.39886, 0.01681, 0.12926), (0.0, 0.0, -0.050815, 0.998708))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a02 = library_object(
            "whitepackerbottle_a02", (0.57243, -0.00667, 0.129), (0.0, 0.0, -0.299802, 0.954002)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon_2 = library_object("spoon_2", (0.49416, 0.24162, 0.13439), (0.0, 0.0, 0.988598, 0.15058))
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.35165, 0.25159, 0.129), (0.0, 0.0, -0.995412, 0.095683)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a02")],
                "container": SceneEntityCfg("av_pan_d"),
                "require_gripper_detached": True,
            },
        )
