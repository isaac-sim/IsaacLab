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
class PickPlace028EnvCfg(VLABenchmarkEnvCfg):
    """Put the tomato sauce can in the black bowl, then the white bottle, on a sofa table."""

    instruction = "put the tomato sauce can in the black bowl, then the white bottle"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_sofatable = library_object(
            "appleseed_sofatable", (0.44969, -0.10898, -0.69682), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.46616, 0.17601, 0.05796), (0.0, 0.0, 0.705341, 0.708868)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.whitepackerbottle_a03 = library_object(
            "whitepackerbottle_a03", (0.58314, -0.00294, 0.058), (0.0, 0.0, -0.047979, 0.998848)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.37389, -0.23337, 0.10902), (0.0, 0.0, 0.783133, 0.621854)
        )
        self.scene.av_plant_succulent_01 = library_object(
            "av_plant_succulent_01", (0.42426, -0.08551, 0.058), (0.0, 0.0, 0.94488, 0.327417)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.52468, -0.1937, 0.09942), (0.0, 0.0, 0.452321, 0.891855)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("tomato_sauce_can"), SceneEntityCfg("whitepackerbottle_a03")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
            },
        )
