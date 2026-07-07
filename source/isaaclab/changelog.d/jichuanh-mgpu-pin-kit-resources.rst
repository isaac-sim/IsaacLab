Changed
^^^^^^^

* Changed the Isaac Lab Kit experiences to use one renderer GPU by default. To
  enable single-process multi-GPU rendering, pass the ``renderer.multiGpu``
  settings explicitly through :class:`~isaaclab.app.AppLauncher`'s ``kit_args``.

Added
^^^^^

* Added ``ISAACLAB_FABRIC_USE_GPU_INTEROP`` to override the corresponding PhysX
  Fabric Kit setting without changing renderer multi-GPU behavior. The multi-GPU
  CI override is a temporary workaround to remove after the underlying Kit/PhysX
  problem is fixed.
