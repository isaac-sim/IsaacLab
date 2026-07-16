Added
^^^^^

* Added ``origin_type``, ``env_index``, ``asset_name``, and ``body_name`` fields to
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`. These replace the removed
  ``ViewerCfg`` / ``ViewportCameraController`` and allow the Kit viewport camera to track a
  world origin, an environment origin, or an asset root / body across simulation steps.
