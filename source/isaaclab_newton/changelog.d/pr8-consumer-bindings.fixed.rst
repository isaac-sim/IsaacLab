* Fixed stale ray origins after manual pose writes by refreshing shared body poses before
  updating the sensor and its targets in the same update.
* Preserved sensor sites and configured frame-transformer target names across hard
  resets, and released native IMU and frame-transform sensors belonging to the old model.
