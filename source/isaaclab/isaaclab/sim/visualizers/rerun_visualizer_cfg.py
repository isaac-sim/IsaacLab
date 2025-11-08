# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Rerun Visualizer."""

from typing import Literal

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class RerunVisualizerCfg(VisualizerCfg):
    """Configuration for Rerun Visualizer.
    
    The Rerun Visualizer integrates with the rerun visualization library, enabling
    real-time or offline visualization with advanced features like time scrubbing
    and data inspection through a web-based interface.
    
    Features:
    - Web-based visualization interface
    - Time scrubbing and playback controls
    - 3D scene navigation
    - Data inspection and filtering
    - Recording and export capabilities
    - Remote viewing support
    
    Note:
        Requires the rerun-sdk package to be installed: pip install rerun-sdk
    """

    # Override defaults for Rerun visualizer
    camera_position: tuple[float, float, float] = (10.0, 10.0, 10.0)
    """Initial position of the camera. Default is (10.0, 10.0, 10.0)."""

    # Rerun-specific settings
    server_mode: bool = True
    """Whether to run in server mode. Default is True.
    
    In server mode, Rerun starts a server that viewers can connect to.
    When False, data is logged to a file or sent to an external viewer.
    """

    server_address: str = "127.0.0.1:9876"
    """Server address for Rerun. Default is "127.0.0.1:9876".
    
    Format: "host:port". Only used when server_mode is True.
    """

    launch_viewer: bool = True
    """Whether to automatically launch the web viewer. Default is True.
    
    When True, the Rerun web viewer will be automatically opened in a browser.
    """

    app_id: str = "isaaclab-simulation"
    """Application identifier for Rerun. Default is "isaaclab-simulation".
    
    This is used to identify the application in the Rerun viewer and for
    organizing recordings.
    """

    recording_path: str | None = None
    """Path to save recordings. Default is None (don't save).
    
    When specified, the Rerun data will be saved to this path for later replay.
    Supported formats: .rrd (Rerun recording format)
    """

    spawn_mode: Literal["connect", "spawn", "save"] = "spawn"
    """How to handle the Rerun viewer. Default is "spawn".
    
    - "connect": Connect to an existing Rerun viewer
    - "spawn": Spawn a new Rerun viewer process
    - "save": Save to a file without opening a viewer
    """

    max_queue_size: int = 100
    """Maximum number of messages to queue. Default is 100.
    
    Controls memory usage for buffering visualization data.
    """

    flush_timeout: float = 2.0
    """Timeout for flushing data to Rerun in seconds. Default is 2.0."""

    log_transforms: bool = True
    """Whether to log rigid body transforms. Default is True."""

    log_meshes: bool = True
    """Whether to log mesh data. Default is True.
    
    When enabled, collision and visual meshes will be logged to Rerun.
    """

    log_cameras: bool = True
    """Whether to log camera data. Default is True."""

    log_point_clouds: bool = False
    """Whether to log point cloud data. Default is False."""

    log_images: bool = False
    """Whether to log images from cameras. Default is False.
    
    Note: Logging images can significantly increase data size and bandwidth.
    """

    log_tensors: bool = False
    """Whether to log tensor data (observations, actions, etc.). Default is False.
    
    When enabled, can log observation buffers, action buffers, and other tensors.
    """

    time_mode: Literal["sim_time", "wall_time", "step_count"] = "sim_time"
    """Time mode for logging. Default is "sim_time".
    
    - "sim_time": Use simulation time as the timeline
    - "wall_time": Use wall clock time
    - "step_count": Use step count as the timeline
    """

    entity_path_prefix: str = "/world"
    """Prefix for entity paths in Rerun. Default is "/world".
    
    All logged entities will be under this prefix in the Rerun hierarchy.
    """

    log_static_once: bool = True
    """Whether to log static scene data only once. Default is True.
    
    When True, static meshes and other unchanging data are logged only at the
    start, reducing data size and bandwidth.
    """

    up_axis: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+z"
    """The up axis for the 3D space. Default is "+z".
    
    This should match your simulation's coordinate system.
    """

    mesh_quality: Literal["low", "medium", "high"] = "medium"
    """Quality level for mesh data. Default is "medium".
    
    Higher quality preserves more detail but increases data size.
    """

    enable_compression: bool = True
    """Whether to enable data compression. Default is True.
    
    Compression reduces bandwidth and storage requirements but adds CPU overhead.
    """

    verbose: bool = False
    """Whether to enable verbose logging. Default is False."""


