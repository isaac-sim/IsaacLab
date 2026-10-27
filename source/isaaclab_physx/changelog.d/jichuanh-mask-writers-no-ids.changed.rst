* Changed the :class:`~isaaclab_physx.assets.Articulation` fixed and spatial tendon mask setters to write the
  tendon buffers through the masks instead of converting them to indices, removing a host wait per call. A mask
  whose length differs from the number of instances or tendons now raises an :class:`AssertionError` unless
  :attr:`~isaaclab.assets.AssetBaseCfg.disable_shape_checks` is set.
