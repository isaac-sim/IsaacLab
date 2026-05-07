Fixed
^^^^^

* Fixed :class:`~isaaclab_ovphysx.physics.OvPhysxManager` passing an
  unsupported ``gpu_index`` parameter to the ``ovphysx.PhysX`` constructor,
  which caused a ``TypeError`` on GPU device initialization.
