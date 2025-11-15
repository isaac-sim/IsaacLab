# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rerun visualizer."""

from __future__ import annotations

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class RerunVisualizerCfg(VisualizerCfg):
    """Configuration for the Rerun visualizer using rerun-sdk.
    
    The Rerun visualizer provides web-based visualization with advanced features:
    
    - Time scrubbing and playback controls
    - 3D scene navigation and inspection
    - Data filtering and analysis
    - Recording to .rrd files for offline replay
    - Built-in timeline and data inspection tools
    
    This visualizer requires the Newton physics backend and the rerun-sdk package:
    
    .. code-block:: bash
    
        pip install rerun-sdk
    
    Note:
        The Rerun visualizer wraps Newton's ViewerRerun, which requires a Newton Model
        and State. It will not work with other physics backends (e.g., PhysX) until
        future support is added.
    
    Example:
        
        .. code-block:: python
        
            from isaaclab.sim.visualizers import RerunVisualizerCfg
            
            visualizer_cfg = RerunVisualizerCfg(
                enabled=True,
                server_mode=True,
                launch_viewer=True,
                keep_historical_data=False,  # Constant memory for training
                record_to_rrd="recording.rrd",  # Save to file
            )
    """

    visualizer_type: str = "rerun"
    """Type identifier for the Rerun visualizer. Defaults to "rerun"."""

    # Connection settings
    server_mode: bool = True
    """Whether to run Rerun in server mode (gRPC). Defaults to True.
    
    If True, Rerun will start a gRPC server that the web viewer connects to.
    If False, data is logged directly (useful for recording to file only).
    """

    server_address: str = "127.0.0.1:9876"
    """Server address and port for gRPC mode. Defaults to "127.0.0.1:9876".
    
    Only used if server_mode=True. The web viewer will connect to this address.
    """

    launch_viewer: bool = True
    """Whether to auto-launch the web viewer. Defaults to True.
    
    If True, a web browser will open showing the Rerun viewer interface.
    The viewer provides 3D visualization, timeline controls, and data inspection.
    """

    app_id: str = "isaaclab-simulation"
    """Application identifier for Rerun. Defaults to "isaaclab-simulation".
    
    This is displayed in the Rerun viewer title and used to distinguish
    multiple Rerun sessions.
    """

    # Data management
    keep_historical_data: bool = False
    """Whether to keep historical transform data in viewer timeline. Defaults to False.
    
    - If False: Only current frame is kept, memory usage is constant (good for training)
    - If True: Full timeline is kept, enables time scrubbing (good for debugging)
    
    For long training runs with many environments, False is recommended to avoid
    memory issues. For analysis and debugging, True allows rewinding the simulation.
    """

    keep_scalar_history: bool = True
    """Whether to keep historical scalar/plot data in viewer. Defaults to True.
    
    Scalar data (plots, metrics) is typically small, so keeping history is
    reasonable even for long runs. This enables viewing plot trends over time.
    """

    # Recording
    record_to_rrd: str | None = None
    """Path to save recording as .rrd file. Defaults to None (no recording).
    
    If specified (e.g., "my_recording.rrd"), all logged data will be saved to
    this file for offline replay and analysis. The file can be opened later with:
    
    .. code-block:: bash
    
        rerun my_recording.rrd
    
    Example paths:
        - "recording.rrd" - saves to current directory
        - "/tmp/recordings/run_{timestamp}.rrd" - absolute path with timestamp
        - None - no recording saved
    """

    # Visualization options
    log_transforms: bool = True
    """Whether to log rigid body transforms. Defaults to True.
    
    Transform logging shows the position and orientation of all rigid bodies
    in the scene. This is the core visualization data.
    """

    log_meshes: bool = True
    """Whether to log mesh geometry. Defaults to True.
    
    Mesh logging shows the 3D shapes of objects. If False, only transforms
    (positions/orientations) are shown as coordinate frames.
    """

    visualize_markers: bool = True
    """Whether to actively log VisualizationMarkers to Rerun. Defaults to True.
    
    If True, markers created via VisualizationMarkers (arrows, frames, spheres, etc.)
    will be converted to Rerun entities and logged each frame. This requires active
    logging (unlike OV visualizer where markers auto-appear in the viewport).
    
    Supported marker types:
        - Arrows (via log_lines)
        - Coordinate frames (via log_lines for XYZ axes)
        - Spheres (via log_points)
    """

    visualize_plots: bool = True
    """Whether to actively log LivePlot data to Rerun. Defaults to True.
    
    If True, scalar data from LivePlots will be logged as time series in Rerun.
    This allows viewing training metrics, rewards, and other scalars alongside
    the 3D visualization.
    
    Note: Currently a stub - full implementation coming soon.
    """

    # Performance and filtering
    max_instances_per_env: int | None = None
    """Maximum number of instances to visualize per environment. Defaults to None (all).
    
    For scenes with many instances per environment, this limits how many are
    visualized to improve performance. None means visualize all instances.
    """

    # Future: PhysX backend support
    # When PhysX support is added, these fields will be used:
    # physics_backend: Literal["newton", "physx"] | None = None
    # """Physics backend to use. Auto-detected if None."""

