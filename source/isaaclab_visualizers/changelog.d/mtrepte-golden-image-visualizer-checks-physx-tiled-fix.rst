Fixed
^^^^^

* Fixed incomplete tiled camera renders for the PhysX physics backend in golden-image and
  tiled-camera integration tests.  :class:`~isaaclab_visualizers.newton.NewtonVisualizer` skipped
  ``_log_camera_sensor_image()`` when the Newton physics state was unavailable (PhysX backend),
  leaving all owned tiled cameras with zero renderer updates during physics warmup; only env 0
  rendered correctly.  The capture helper now pumps ``camera_sensor.update()`` for
  ``_TILED_CAMERA_SENSOR_WARMUP_UPDATES`` iterations before sampling, matching the warmup already
  applied to Kit viewport and Newton viewer paths.

* Fixed :class:`~isaaclab_visualizers.kit.KitVisualizer` silently skipping tiled camera sensor
  creation in headless mode even when ``--enable_cameras`` is active.  The camera sensor is now
  always created when camera rendering is available; only the interactive UI image window is
  suppressed in headless mode.
