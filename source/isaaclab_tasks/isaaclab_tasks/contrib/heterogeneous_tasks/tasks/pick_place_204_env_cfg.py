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
class PickPlace204EnvCfg(VLABenchmarkEnvCfg):
    """Place both the plastic bottle and the barbecue sauce bottle into the metal bin, on an office desk."""

    instruction = "place both the plastic bottle and the barbecue sauce bottle into the metal bin"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.50381, 0.44344, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.43853, -0.08552, 0.007), (0.0, 0.0, 0.44623, 0.894918))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.spaghetti = library_object("spaghetti", (0.39436, 0.17435, 0.03192), (0.0, 0.0, 0.439081, 0.898447))
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.naturalbostonroundbottle_a02 = library_object(
            "naturalbostonroundbottle_a02", (0.48577, 0.24501, 0.007), (0.0, 0.0, 0.624331, 0.78116)
        )
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.34862, 0.07014, 0.00706), (0.0, 0.0, -0.797176, 0.603747)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("naturalbostonroundbottle_a02"), SceneEntityCfg("bbq_sauce_bottle_hot3d")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
