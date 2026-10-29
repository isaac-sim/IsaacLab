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
class PickPlace082EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the tomato sauce can and put it in the black bowl, on a conference table."""

    instruction = "pick up the tomato sauce can and put it in the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.cline = library_object("cline", (0.75781, 0.16442, -0.6971), (0.0, 0.0, 1.0, 0.0), kinematic=True)
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.39277, 0.02993, 0.12896), (0.0, 0.0, -0.071428, 0.997446)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.crabbypenholder = library_object(
            "crabbypenholder", (0.37888, -0.16737, 0.12897), (0.0, 0.0, 0.195796, 0.980645)
        )
        self.scene.oatmeal_raisin_cookies = library_object(
            "oatmeal_raisin_cookies", (0.49158, 0.19122, 0.21262), (0.0, 0.0, 0.041536, 0.999137)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.55348, 0.09717, 0.191), (0.0, 0.0, -0.044126, 0.999026)
        )
        self.scene.storage_box = library_object(
            "storage_box", (0.47967, -0.16925, 0.12887), (0.0, 0.0, 0.635671, 0.77196)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.36558, -0.24321, 0.17042), (0.0, 0.0, 0.709727, 0.704477)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("tomato_sauce_can")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
