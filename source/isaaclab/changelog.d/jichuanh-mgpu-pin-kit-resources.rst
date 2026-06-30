Changed
^^^^^^^

* Changed :class:`~isaaclab.app.AppLauncher` to disable single-process multi-GPU
  rendering by default. Set ``multi_gpu=True`` or pass ``--multi_gpu`` to restore
  the previous rendering behavior.

Added
^^^^^

* Added ``ISAACLAB_FABRIC_USE_GPU_INTEROP`` to override the corresponding PhysX
  Fabric Kit setting without changing renderer multi-GPU behavior. The multi-GPU
  CI override is a temporary workaround to remove after the underlying Kit/PhysX
  problem is fixed.
