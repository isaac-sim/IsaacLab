Changed
^^^^^^^

* Changed :class:`~isaaclab_visualizers.kit.KitVisualizer` to leave its viewport
  camera unpartitioned when AppLauncher enables the all-environment spectator
  view. Otherwise, the viewport camera is assigned to the first visible
  environment.

Fixed
^^^^^

* Fixed global Kit/USD visualization-marker instances appearing across tiled
  environments when per-instance environment IDs are provided.
