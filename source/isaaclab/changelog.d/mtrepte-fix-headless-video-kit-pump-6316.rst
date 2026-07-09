Fixed
^^^^^

* Fixed headless video recording (``--video`` / ``rgb_array``) forcing a Kit ``app.update()``
  on every environment step. :attr:`~isaaclab.sim.SimulationContext.is_rendering` no longer
  reports offscreen rendering as continuous rendering, so Kit is now pumped on demand only when
  a frame is actually requested. GUI, RTX sensor, visualizer, and XR rendering are unaffected.
