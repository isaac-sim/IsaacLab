Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.physics.OvPhysxManager` refusing to start a CPU simulation after a CUDA simulation, or
  the reverse, in the same process. CPU scenes selected CPU dynamics through their physics-scene attributes instead of
  the OVPhysX process-wide CPU-only mode, which cannot be reverted. To keep OVPhysX off CUDA entirely, set
  ``OVPHYSX_DISABLE_GPU``.
