# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..camera.tiled_camera_cfg import TiledCameraCfg
from ..sensor_base_cfg import SensorBaseCfg
from .visuotactile_sensor import VisuoTactileSensor

##
# GelSight Render Configuration
##


@configclass
class GelSightRenderCfg:
    """Configuration for GelSight sensor rendering parameters.

    This configuration defines the rendering parameters for example-based tactile image synthesis
    using the Taxim approach. 

    Reference:
        Si, Z., & Yuan, W. (2022). Taxim: An example-based simulation model for GelSight
        tactile sensors. IEEE Robotics and Automation Letters, 7(2), 2361-2368.
        https://arxiv.org/abs/2109.04027

    Data Directory Structure:
        The sensor data should be organized in the following structure::

            base_data_path/
            └── sensor_data_dir_name/
                ├── bg.jpg              # Background image (required)
                ├── polycalib.npz       # Polynomial calibration data (required)
                └── real_bg.npy         # Real background data (optional)

    Path Resolution:
        - If ``base_data_path`` is ``None`` (default): Downloads from Isaac Lab Nucleus at
          ``{ISAACLAB_NUCLEUS_DIR}/TacSL/{sensor_data_dir_name}/{filename}``
        - If ``base_data_path`` is provided: Uses custom path at
          ``{base_data_path}/{sensor_data_dir_name}/{filename}``

    Example:
        Using predefined sensor configuration::

            from isaaclab_assets.sensors import GELSIGHT_R15_CFG
            sensor_cfg = VisuoTactileSensorCfg(render_cfg=GELSIGHT_R15_CFG)

        Using custom sensor data::

            custom_cfg = GelSightRenderCfg(
                base_data_path="/path/to/my/sensors",
                sensor_data_dir_name="my_custom_sensor",
                image_height=480,
                image_width=640,
                mm_per_pixel=0.05,
            )
    """

    base_data_path: str | None = None
    """Base path to the directory containing sensor calibration data.

    If ``None`` (default), downloads and caches data from Isaac Lab Nucleus directory.
    If a custom path is provided, uses the data directly from that location without copying.
    """

    sensor_data_dir_name: str = "gelsight_r15_data"
    """Directory name containing the sensor calibration and background data."""

    background_path: str = "bg.jpg"
    """Filename of the background image within the data directory."""

    calib_path: str = "polycalib.npz"
    """Filename of the polynomial calibration data within the data directory."""

    real_background: str = "real_bg.npy"
    """Filename of the real background data within the data directory."""

    image_height: int = 320
    """Height of the tactile image in pixels."""

    image_width: int = 240
    """Width of the tactile image in pixels."""

    num_bins: int = 120
    """Number of bins for gradient magnitude and direction quantization."""

    mm_per_pixel: float = 0.0877
    """Millimeters per pixel conversion factor for the tactile sensor."""


##
# Visuo-Tactile Sensor Configuration
##


@configclass
class VisuoTactileSensorCfg(SensorBaseCfg):
    """Configuration for the visuo-tactile sensor.

    This sensor provides both camera-based tactile sensing and force field tactile sensing.
    It can capture tactile RGB/depth images and compute penalty-based contact forces.
    """

    class_type: type = VisuoTactileSensor

    # Sensor type and capabilities
    render_cfg: GelSightRenderCfg = GelSightRenderCfg()
    """Configuration for GelSight sensor rendering.

    This defines the rendering parameters for converting depth maps to realistic tactile images.
    Defaults to GelSight R1.5 parameters. Use predefined configs like GELSIGHT_R15_CFG or
    GELSIGHT_MINI_CFG from isaaclab_assets.sensors for standard sensor models.
    """

    enable_camera_tactile: bool = True
    """Whether to enable camera-based tactile sensing."""

    enable_force_field: bool = True
    """Whether to enable force field tactile sensing."""

    # Elastomer configuration
    elastomer_rigid_body: str = "elastomer"
    """Prim path of the elastomer rigid body for tactile sensing."""

    elastomer_tactile_mesh: str = "elastomer/visual"
    """Prim path of the elastomer mesh for tactile point generation."""

    elastomer_tip_link_name: str = "elastomer_tip"
    """Prim path of the elastomer tip link."""

    # Force field configuration
    num_tactile_rows: int = 20
    """Number of rows of tactile points for force field sensing."""

    num_tactile_cols: int = 25
    """Number of columns of tactile points for force field sensing."""

    tactile_margin: float = 0.003
    """Margin for tactile point generation (in meters)."""

    # Indenter configuration for force field sensing
    indenter_rigid_body: str | None = None
    """Prim path of the indenter rigid body for SDF-based collision detection."""

    indenter_sdf_mesh: str | None = None
    """Prim path of the indenter SDF mesh for SDF-based collision detection."""

    # Force field physics parameters
    tactile_kn: float = 1.0
    """Normal contact stiffness for penalty-based force computation."""

    tactile_mu: float = 2.0
    """Friction coefficient for shear forces."""

    tactile_kt: float = 0.1
    """Tangential stiffness for shear forces."""

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
