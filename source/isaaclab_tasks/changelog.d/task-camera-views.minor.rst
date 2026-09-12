Changed
^^^^^^^

* Defined fixed recording views for static core tasks and tracking views for locomotion, with
  heading smoothing of 0.2 seconds (0.5 for Ant and Humanoid). Explicit visualizer settings
  override task defaults; set ``origin_heading_smoothing_time_constant=0.0`` for immediate yaw.
