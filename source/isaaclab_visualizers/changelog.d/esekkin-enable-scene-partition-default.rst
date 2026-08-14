Changed
^^^^^^^

* Changed :class:`~isaaclab_visualizers.kit.KitVisualizer` to author the
  viewport camera's scene-partition attribute by default. Set
  ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0`` to disable it.

Fixed
^^^^^

* Fixed global Kit/USD visualization-marker instances appearing across tiled
  environments when per-instance environment IDs are provided.
