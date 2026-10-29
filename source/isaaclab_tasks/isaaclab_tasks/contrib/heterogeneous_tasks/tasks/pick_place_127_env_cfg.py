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
class PickPlace127EnvCfg(VLABenchmarkEnvCfg):
    """Place the spoon into the black bowl, on a table."""

    instruction = "place the spoon into the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_oak = library_object(
            "table_oak", (0.58519, 0.05788, -0.347), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.43223, 0.03831, 0.00894), (0.0, 0.0, -0.643293, 0.76562)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.orange_02 = library_object("orange_02", (0.31468, -0.2183, 0.00906), (0.0, 0.0, -0.991932, 0.126772))
        self.scene.pumpkinsmall = library_object(
            "pumpkinsmall", (0.52777, -0.21736, 0.00874), (0.0, 0.0, -0.133856, 0.991001)
        )
        self.scene.spoon_2 = library_object("spoon_2", (0.36764, 0.21831, 0.01439), (0.0, 0.0, 0.151244, 0.988496))
        self.scene.yellow_block = library_object(
            "yellow_block", (0.40266, -0.16798, 0.0325), (0.0, 0.0, -0.004301, 0.999991)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_2")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "require_gripper_detached": True,
            },
        )
