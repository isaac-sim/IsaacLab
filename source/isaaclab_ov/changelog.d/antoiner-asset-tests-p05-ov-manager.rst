Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.physics.OvPhysxManager` refusing to start a CPU simulation after a CUDA simulation, or
  the reverse, in the same process.

Changed
^^^^^^^

* Changed CPU simulations to select CPU dynamics through the attributes of every physics scene in the stage instead
  of the OVPhysX process-wide CPU-only mode, which cannot be reverted. To keep OVPhysX off CUDA entirely, set
  ``OVPHYSX_DISABLE_GPU``.
