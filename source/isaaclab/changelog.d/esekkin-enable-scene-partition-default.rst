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
  still supplies the config construction default and now accepts only ``0`` or
  ``1``. OVRTX remains always-on.

Fixed
^^^^^

* Fixed built-in pose and velocity command-marker instances appearing across
  environment scene partitions.
