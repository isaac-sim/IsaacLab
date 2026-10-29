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
class PickPlace177EnvCfg(VLABenchmarkEnvCfg):
    """Place both the yogurt cup and the mustard bottle into the red bowl, on an office desk."""

    instruction = "place both the yogurt cup and the mustard bottle into the red bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contemporary_lowfulldesk = library_object(
            "contemporary_lowfulldesk", (0.49383, -0.04208, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.5007, 0.01951, 0.03448), (0.0, 0.0, -0.992608, 0.121368))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.42019, -0.13196, 0.03384), (0.0, 0.0, 0.389686, 0.920948)
        )
        self.scene.milkjug_a02 = library_object(
            "milkjug_a02", (0.52909, -0.17414, 0.007), (0.0, 0.0, 0.563584, 0.826059)
        )
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.35666, -0.23022, 0.02366), (0.0, 0.0, 0.443551, 0.896249)
        )
        self.scene.naturalbostonroundbottle_a03 = library_object(
            "naturalbostonroundbottle_a03", (0.30774, 0.12221, 0.007), (0.0, 0.0, -0.153965, 0.988076)
        )
        self.scene.mustard_bottle = library_object(
            "mustard_bottle", (0.37175, 0.22365, 0.08703), (0.0, 0.0, -0.121475, 0.992594)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("yogurt_cup"), SceneEntityCfg("mustard_bottle")],
                "container": SceneEntityCfg("bowl_ycb"),
                "require_gripper_detached": True,
            },
        )
