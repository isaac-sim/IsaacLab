Changed
^^^^^^^

* Changed :class:`~isaaclab_visualizers.kit.KitVisualizer` to leave its
  viewport camera unpartitioned for the default all-environment spectator view.
  When ``show_all_partitions_by_default`` is disabled, the viewport camera is
  assigned to the first visible environment.

Fixed
^^^^^

* Fixed global Kit/USD visualization-marker instances appearing across tiled
  environments when per-instance environment IDs are provided.
