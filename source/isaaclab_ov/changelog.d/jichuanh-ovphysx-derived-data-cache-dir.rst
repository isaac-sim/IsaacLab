Added
^^^^^

* Added :attr:`~isaaclab_ov.physics.OvPhysxCfg.cooked_collider_cache_dir` to select where OVPhysX
  writes its cooked-collider cache. It defaults to a per-user directory under the system temporary
  directory, and cooked colliders now persist across runs.

Fixed
^^^^^

* Fixed OvPhysX writing its cooked-collider cache into the directory holding the Python interpreter,
  which logged ``omni.datastore`` errors when that directory was not writable.
