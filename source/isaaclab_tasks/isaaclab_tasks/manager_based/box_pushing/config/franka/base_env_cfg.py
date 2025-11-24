# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.box_pushing.box_pushing_env_cfg import BoxPushingEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class FrankaBoxPushingBaseEnvCfg(BoxPushingEnvCfg):
    """Shared scene setup for Franka box pushing environments.

    Sub-classes must provide ``robot_cfg_r2r``/``robot_cfg_comparison`` and implement ``_configure_actions``.
    """

    robot_cfg_r2r: ArticulationCfg = MISSING
    robot_cfg_comparison: ArticulationCfg = MISSING

    def __post_init__(self):
        super().__post_init__()
        self._configure_robot()
        self._configure_scene_objects()
        self._configure_actions()

    def _configure_robot(self):
        if self.robot_cfg_r2r is MISSING or self.robot_cfg_comparison is MISSING:
            raise ValueError("Robot configs must be provided by subclass.")
        # ensure prim paths are set to satisfy config validation before using the configs
        self.robot_cfg_r2r = self.robot_cfg_r2r.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.robot_cfg_comparison = self.robot_cfg_comparison.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot = self.robot_cfg_r2r if self.r2r else self.robot_cfg_comparison

    def _configure_scene_objects(self):
        # command target body
        self.commands.object_pose.body_name = "panda_hand"

        # box to push
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, 0.0, 0.01) if self.r2r else (0.4, 0.3, -0.01),
                rot=(1, 0, 0, 0) if self.r2r else (0, 0, 0, 1),
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/box.usda"),
                scale=(0.001, 0.001, 0.001),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # end-effector frame visualization
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.27],
                    ),
                ),
            ],
        )

    def _configure_actions(self):
        """Implemented by subclasses to set action term configuration."""
        raise NotImplementedError
