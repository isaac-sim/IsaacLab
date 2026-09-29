Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.physics.OvPhysxManager` refusing to start a CPU simulation after a CUDA one, or the
  reverse, in the same process. CPU scenes now select CPU dynamics through their physics-scene attributes instead of
  the OVPhysX process-wide CPU-only mode, which cannot be reverted. Deployments that must keep OVPhysX off CUDA can
  set ``OVPHYSX_DISABLE_GPU``.
