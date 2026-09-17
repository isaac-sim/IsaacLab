# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import REQUIRED, replace_config
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from .pose_commands import CableUniformPoseCommand, DeformableUniformPoseCommand, ObjectUniformPoseCommand

ALIGN_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
        "position_far": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "position_near": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


@dataclass
class ObjectUniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type["ObjectUniformPoseCommand"] | str = (
        "isaaclab_tasks.core.lift.mdp.commands.pose_commands:ObjectUniformPoseCommand"
    )

    asset_name: str = REQUIRED
    """Name of the coordinate referencing asset in the environment for which the commands are generated respect to."""

    object_name: str = REQUIRED
    """Name of the object in the environment for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @dataclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = REQUIRED
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = REQUIRED
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = REQUIRED
        """Range for the z position (in m)."""

        roll: tuple[float, float] = REQUIRED
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = REQUIRED
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = REQUIRED
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = REQUIRED
    """Ranges for the commands."""

    position_only: bool = True
    """Command goal position only. Command includes goal quat if False"""

    # Pose Markers
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = field(
        default_factory=lambda: replace_config(ALIGN_MARKER_CFG, prim_path="/Visuals/Command/goal_pose")
    )
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = field(
        default_factory=lambda: replace_config(ALIGN_MARKER_CFG, prim_path="/Visuals/Command/body_pose")
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    success_vis_asset_name: str = REQUIRED
    """Name of the asset in the environment for which the success color are indicated."""

    # success markers
    success_visualizer_cfg: VisualizationMarkersCfg = field(
        default_factory=lambda: VisualizationMarkersCfg(prim_path="/Visuals/SuccessMarkers", markers={})
    )
    """The configuration for the success visualization marker. User needs to add the markers"""


@dataclass
class DeformableUniformPoseCommandCfg(ObjectUniformPoseCommandCfg):
    """Configuration for the deformable uniform pose command generator."""

    class_type: type["DeformableUniformPoseCommand"] | str = (
        "isaaclab_tasks.core.lift.mdp.commands.pose_commands:DeformableUniformPoseCommand"
    )


@dataclass
class CableUniformPoseCommandCfg(ObjectUniformPoseCommandCfg):
    """Configuration for a cable segment uniform pose command generator."""

    class_type: type["CableUniformPoseCommand"] | str = (
        "isaaclab_tasks.core.lift.mdp.commands.pose_commands:CableUniformPoseCommand"
    )

    segment_index: int = REQUIRED
    """Zero-based cable segment index tracked by the command."""
