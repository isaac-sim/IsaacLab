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
class PickPlace104EnvCfg(VLABenchmarkEnvCfg):
    """Put the barbecue sauce bottle in the metal bin, on a table."""

    instruction = "put the barbecue sauce bottle in the metal bin"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.58619, -0.00783, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.46854, -0.03424, 0.009), (0.0, 0.0, -0.837588, 0.546303))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_17 = library_object(
            "spatula_17", (0.32837, 0.07619, 0.01268), (0.0, 0.0, 0.748109, 0.663576)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.smartphone = library_object(
            "smartphone", (0.48186, -0.21385, 0.01465), (0.0, 0.0, -0.834162, 0.551519)
        )
        self.scene.spoon_1 = library_object("spoon_1", (0.35059, -0.17891, 0.01966), (0.0, 0.0, 0.763447, 0.645871))
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.45048, 0.19764, 0.00896), (0.0, 0.0, -0.453721, 0.891144)
        )
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.56734, 0.08946, 0.08316), (0.0, 0.0, 0.999927, 0.012086)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("bbq_sauce_bottle")],
                "container": SceneEntityCfg("bin_a03"),
                "require_gripper_detached": True,
            },
        )
