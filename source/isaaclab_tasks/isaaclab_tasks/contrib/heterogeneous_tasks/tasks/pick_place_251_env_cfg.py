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
class PickPlace251EnvCfg(VLABenchmarkEnvCfg):
    """Take the crab pen holder out of the white bowl, on a hexagonal table."""

    instruction = "take the crab pen holder out of the white bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_hexagon = library_object(
            "av_hexagon", (0.74307, 0.00114, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.41022, 0.04277, 0.0708), (0.0, 0.0, 0.998688, 0.051208)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.41036, 0.04276, 0.08497), (0.0, 0.0, -0.802611, 0.596503)
        )
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.granola_bars = library_object(
            "granola_bars", (0.53579, -0.15087, 0.133), (0.0, 0.0, -0.366807, 0.930297)
        )
        self.scene.measuring_cups_1 = library_object(
            "measuring_cups_1", (0.43105, 0.23145, 0.09703), (0.0, 0.0, -0.000543, 1.0)
        )
        self.scene.clamp = library_object("clamp", (0.3563, -0.17158, 0.09058), (0.0, 0.0, -0.715976, 0.698125))
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("crabbypenholder")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
