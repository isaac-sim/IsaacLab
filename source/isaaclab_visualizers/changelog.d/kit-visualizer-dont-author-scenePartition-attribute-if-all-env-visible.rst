Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.kit.KitVisualizer` to not author the ``omni:scenePartition``
  attribute onto the viewport camera when all environments are visible. Previously, the attribute
  was authored with the value "env_0", causing only environment 0 to be visualized.
