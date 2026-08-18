Added
^^^^^

* Added an ``environment_ids`` argument to
  :meth:`~isaaclab.markers.VisualizationMarkers.visualize` for assigning marker
  instances to renderer scene partitions.

Changed
^^^^^^^

* Changed per-environment Isaac RTX scene partitioning to be enabled by default
  through ``IsaacRtxRendererCfg.enable_scene_partitioning``. The legacy
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION`` environment variable
  still supplies the construction default and now accepts only ``0`` or ``1``.
  Set ``IsaacRtxRendererCfg(enable_scene_partitioning=False)`` to preserve the
  previous unpartitioned behavior. OVRTX remains always-on.
* Changed :class:`~isaaclab.app.AppLauncher` to initialize the all-environment
  spectator view when the Kit viewport is enabled or Kit visualization, recording,
  livestreaming, or XR is requested. Regular headless training and camera-sensor
  runs retain partition isolation.
* Changed :meth:`~isaaclab.markers.VisualizationMarkers.visualize` to reject
  per-marker input arrays with differing first dimensions. Pass one entry per
  marker in every supplied array.

Fixed
^^^^^

* Fixed built-in pose and velocity command-marker instances appearing across
  environment scene partitions.
