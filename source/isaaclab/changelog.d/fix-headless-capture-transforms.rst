Added
^^^^^

* Added :meth:`~isaaclab.sim.SimulationContext.pre_render` to publish deferred physics state
  (e.g. Newton's Fabric/USD transform sync) to the rendering backend without a full
  :meth:`~isaaclab.sim.SimulationContext.render`, so a single consumer can refresh its transforms
  without stepping every other visualizer or firing every registered render callback.
* Added :meth:`~isaaclab.sim.SimulationContext.refresh_visualizer` to step one visualizer (and
  dispatch marker/live-plot callbacks it consumes) without stepping any other registered
  visualizer. Some visualizers (e.g. ``NewtonGLVisualizer`` / ``NewtonRTXVisualizer``) cache
  their own render-ready state and only refresh it in ``step()``, so publishing transforms via
  ``pre_render()`` alone is not enough for those to see fresh data on the next capture.

Fixed
^^^^^

* Fixed on-demand visualizer recording to publish backend transforms before frame capture,
  preventing stale Newton poses in headless Kit videos. Capture now calls
  :meth:`~isaaclab.sim.SimulationContext.forward`, the new
  :meth:`~isaaclab.sim.SimulationContext.pre_render`, and the new
  :meth:`~isaaclab.sim.SimulationContext.refresh_visualizer` directly instead of a full
  :meth:`~isaaclab.sim.SimulationContext.render`, avoiding double-rendering unrelated headless
  visualizers or invalidating another recorder's cached frame within the same physics step.
* Fixed on-demand capture of the Newton GL/RTX visualizers (``source='visualizer:newton_gl'`` /
  ``'visualizer:newton_rtx'``) to also refresh their cached render state before reading a frame;
  it was previously refreshed only at the render cadence via ``update_visualizers()``.
* :meth:`~isaaclab.sim.SimulationContext.render` now calls the new
  :meth:`~isaaclab.sim.SimulationContext.pre_render` wrapper instead of duplicating the
  ``physics_manager.pre_render()`` call.
* Documented the render-state refresh required by custom on-demand visualizer recorders.
