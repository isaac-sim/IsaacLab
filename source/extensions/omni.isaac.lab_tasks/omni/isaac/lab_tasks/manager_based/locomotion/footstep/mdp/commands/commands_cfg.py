# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


from .footstep_pose_command import FootstepPoseCommand


@configclass
class FootstepPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`FootstepPoseCommand` class for more details.
    """

    class_type: type = FootstepPoseCommand
    # resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    # init_pos_offset: tuple[float, float] = (0.0, 0.0)
    # """Init footstep pose
    # """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the footstep position x command (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the footstep position y command (in m)."""

        ang_z: tuple[float, float] = MISSING
        """Range for the footstep orientation along z-axis command (in rad)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the footstep pose commands."""

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    angle_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    position_success_threshold: float = MISSING
    """Threshold for the position error to consider the goal position to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    goal_pose_visualizer_left_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker_left",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.092, 0.1576, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )

    goal_pose_visualizer_right_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker_right",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.092, 0.1576, 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )
    