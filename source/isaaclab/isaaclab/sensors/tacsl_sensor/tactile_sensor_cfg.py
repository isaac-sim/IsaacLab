# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the tactile sensor."""

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..camera.tiled_camera_cfg import TiledCameraCfg
from ..sensor_base_cfg import SensorBaseCfg
from .tactile_sensor import TactileSensor


@configclass
class TactileSensorCfg(SensorBaseCfg):
    """Configuration for the tactile sensor.

    This sensor provides both camera-based tactile sensing and force field tactile sensing.
    It can capture tactile RGB/depth images and compute penalty-based contact forces.
    """

    class_type: type = TactileSensor

    # Sensor type and capabilities
    sensor_type: str = "gelsight_r15"
    """Type of tactile sensor. Options: 'gelsight_r15', 'gs_mini'."""

    enable_camera_tactile: bool = True
    """Whether to enable camera-based tactile sensing."""

    enable_force_field: bool = True
    """Whether to enable force field tactile sensing."""

    # Elastomer configuration
    elastomer_link_name: str = "elastomer"
    """Name of the elastomer link for tactile sensing."""

    elastomer_tip_link_name: str = "elastomer_tip"
    """Name of the elastomer tip link."""

    # Force field configuration
    num_tactile_rows: int = 20
    """Number of rows of tactile points for force field sensing."""

    num_tactile_cols: int = 25
    """Number of columns of tactile points for force field sensing."""

    tactile_margin: float = 0.003
    """Margin for tactile point generation (in meters)."""

    # Indenter configuration for force field sensing
    indenter_actor_name: str | None = None
    """Name of the indenter actor for SDF-based collision detection."""

    indenter_link_name: str | None = None
    """Name of the indenter link for rigid body dynamics."""

    indenter_mesh_name: str | None = None
    """Name of the indenter mesh for SDF-based collision detection."""

    indenter_urdf_path: str | None = None
    """Path to the indenter URDF file (required for trimesh/pysdf tools)."""

    # Force field physics parameters
    tactile_kn: float = 1.0
    """Normal contact stiffness for penalty-based force computation."""

    tactile_damping: float = 0.003
    """Damping coefficient for contact forces."""

    tactile_mu: float = 2.0
    """Friction coefficient for shear forces."""

    tactile_kt: float = 0.1
    """Tangential stiffness for shear forces."""

    # Compliant dynamics configuration
    compliance_stiffness: float = 1.0
    """Compliance stiffness for elastomer dynamics."""

    compliant_damping: float = 0.1
    """Compliant damping for elastomer dynamics."""

    elastomer_collision_path: str = "collisions/compliant_contact"
    """Path to the elastomer collision geometry."""

    # Camera configuration (optional, for camera-based tactile sensing)
    camera_cfg: TiledCameraCfg | None = None
    """Camera configuration for tactile RGB/depth sensing.

    If None, camera-based sensing will be disabled even if enable_camera_tactile is True.
    """

    # Visualization
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(prim_path="/Visuals/TactileSensor")
    """The configuration object for the visualization markers.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """

    trimesh_vis_tactile_points: bool = False
    """Whether to visualize tactile points for debugging using trimesh."""
    visualize_sdf_closest_pts: bool = False
    """Whether to visualize SDF closest points for debugging."""
