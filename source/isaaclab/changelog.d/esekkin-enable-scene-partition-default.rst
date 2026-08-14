Added
^^^^^

* Added an ``environment_ids`` argument to
  :meth:`~isaaclab.markers.VisualizationMarkers.visualize` for assigning marker
  instances to renderer scene partitions.

Changed
^^^^^^^

* Changed per-environment Isaac RTX scene partitioning to be enabled by default. Set
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0`` to disable Isaac RTX
  scene-partition authoring and Kit viewport-camera tagging. OVRTX remains always-on.
  The environment variable now accepts only ``0`` or ``1``.
