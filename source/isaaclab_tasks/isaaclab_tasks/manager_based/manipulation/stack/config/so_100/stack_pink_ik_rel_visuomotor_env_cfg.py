# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp

from ... import mdp
from .stack_pink_ik_rel_env_cfg import SO100CubeStackPinkIKRelEnvCfg, ObservationsCfg


@configclass
class VisuomotorObservationsCfg(ObservationsCfg):
    """Observation specifications for the visuomotor MDP."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group with state values and camera."""

        table_cam = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SO100CubeStackPinkIKRelVisuomotorEnvCfg(SO100CubeStackPinkIKRelEnvCfg):
    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss
        