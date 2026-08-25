Added
^^^^^

* Added :class:`~isaaclab.cloner.selection_utils.SceneEntitySelectionCfg`, a heterogeneous extension
  of :class:`~isaaclab.managers.SceneEntityCfg` that maps global environment IDs to the instance rows
  of a partially populated asset physics view and scatters view-ordered values back into global
  environment order. Only PhysX is supported because the mapping is read from the view's ``prim_paths``
  metadata.
