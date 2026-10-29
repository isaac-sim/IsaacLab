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
class PickPlace064EnvCfg(VLABenchmarkEnvCfg):
    """Put the wooden box and the raisin box in the black bowl, on a table."""

    instruction = "put the wooden box and the raisin box in the black bowl"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.91172, 0.09635, -0.697), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.44637, -0.15692, -0.05804), (0.0, 0.0, -0.977049, 0.213015)
        )
        self.scene.av_coaster_hexagon = library_object(
            "av_coaster_hexagon", (0.5067, 0.12702, -0.058), (0.0, 0.0, -0.966413, 0.256994)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.cream_cheese = library_object(
            "cream_cheese", (0.39075, 0.23784, -0.0314), (0.0, 0.0, 0.999452, 0.033105)
        )
        self.scene.raisin_box = library_object(
            "raisin_box", (0.28626, 0.11598, 0.00359), (0.0, 0.0, -0.153186, 0.988197)
        )
        self.scene.storage_box = library_object(
            "storage_box", (0.38826, 0.07992, -0.05813), (0.0, 0.0, -0.907839, 0.41932)
        )
        self.scene.dry_erase_marker = library_object(
            "dry_erase_marker", (0.30231, 0.23848, -0.04856), (0.0, 0.0, 0.286167, 0.95818)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("storage_box"), SceneEntityCfg("raisin_box")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
