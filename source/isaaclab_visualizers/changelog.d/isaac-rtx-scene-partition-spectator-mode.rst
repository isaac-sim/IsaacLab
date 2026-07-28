Fixed
^^^^^

* Fixed the perspective camera in the kit rendering app showing only the first
  environment when RTX scene partitioning is enabled. Added
  ``rtx.scenePartitioning.showAllPartitionsByDefault = true`` to
  ``isaaclab.python.rendering.kit`` so all environments are visible by default.
* Fixed :class:`~isaaclab_visualizers.kit.KitVisualizer` incorrectly writing a
  scene-partition attribute onto the camera prim in spectator mode (when no
  specific environments are selected). The camera partition update now returns
  early when ``_resolved_visible_env_ids`` is ``None``.
