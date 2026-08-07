Added
^^^^^

* Added :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg` (``--viz newton_rtx``) — an
  OVRTX path-tracer backend with studio lighting.  Shares the full Isaac Lab HUD with
  :class:`~isaaclab_visualizers.newton.NewtonGLVisualizer` via a common mixin.
  Visualization markers, live plots, and the streaming camera panel are not yet supported
  in this release.
* Added streaming camera panel to all visualizers (Newton GL, Kit, Rerun, Viser) via the
  ``streaming_view=True`` option on :class:`~isaaclab.visualizers.VisualizerCfg`.  The panel
  composites per-environment camera frames for RGB, depth (turbo colormap), segmentation
  (golden-ratio palette), and surface normals.

Changed
^^^^^^^

* :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg` now defaults
  ``streaming_view=True`` so the streaming camera panel is active without explicit
  configuration.
* Newton GL sidebar layout: renamed the **Isaac Lab** section to **Simulation** and promoted
  **Streaming View** to a standalone top-level section (same level as Simulation, Live Plots,
  and Visualization Markers).  The section now opens expanded by default and contains a
  **Toggle** combo (open/hide the floating panel) and a **Source Camera** dropdown to switch
  between scene cameras at runtime.
* The streaming camera panel floating window now opens sized to the actual composite
  aspect ratio of the grid (e.g. 4:3 for 12 environments) rather than always square.
* Added runtime camera-selector dropdown (**Source Camera**) to the Newton GL streaming view,
  allowing users to switch between scene cameras (e.g. ``base_camera`` / ``wrist_camera``)
  without restarting the simulation.
* Renamed :class:`~isaaclab_visualizers.newton.NewtonVisualizer` to
  :class:`~isaaclab_visualizers.newton.NewtonGLVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg` to
  :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg`.  The old names are kept as
  deprecated aliases.

Deprecated
^^^^^^^^^^

* :class:`~isaaclab_visualizers.newton.NewtonVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg` are deprecated in favor of
  :class:`~isaaclab_visualizers.newton.NewtonGLVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg`.
