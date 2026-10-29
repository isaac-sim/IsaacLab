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
class PickPlace060EnvCfg(VLABenchmarkEnvCfg):
    """First put the kiwi in the red bowl, then put the yellow block in it, on a console table."""

    instruction = "first put the kiwi in the red bowl, then put the yellow block in it"
    episode_length_s = 105.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.40461, -0.08349, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl_ycb = library_object("bowl_ycb", (0.44304, 0.06847, 0.22548), (0.0, 0.0, -0.516613, 0.856219))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.29421, 0.19026, 0.19799), (0.0, 0.0, -0.969501, 0.245089))
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.32766, -0.14114, 0.198), (0.0, 0.0, 0.865782, 0.500421)
        )
        self.scene.fork_big = library_object("fork_big", (0.40842, -0.11511, 0.19835), (0.0, 0.0, 0.960565, 0.278055))
        self.scene.spoon_1 = library_object("spoon_1", (0.37922, 0.24162, 0.20866), (0.0, 0.0, 0.950143, 0.311815))
        self.scene.yellow_block = library_object(
            "yellow_block", (0.32847, 0.00534, 0.2215), (0.0, 0.0, -0.833348, 0.552749)
        )
        self.scene.avocado01 = library_object("avocado01", (0.29602, -0.23763, 0.19783), (0.0, 0.0, -0.133122, 0.9911))
        self.terminations.success = DoneTerm(
            func=mdp.objects_placed_in_container_in_order,
            params={
                "objects": [SceneEntityCfg("av_kiwi01"), SceneEntityCfg("yellow_block")],
                "container": SceneEntityCfg("bowl_ycb"),
            },
        )
