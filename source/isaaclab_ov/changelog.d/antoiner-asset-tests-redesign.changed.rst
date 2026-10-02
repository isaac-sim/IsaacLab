* Changed CPU simulations to select CPU dynamics through the attributes of every physics scene in the stage instead
  of the OVPhysX process-wide CPU-only mode, which cannot be reverted. To keep OVPhysX off CUDA entirely, set
  ``OVPHYSX_DISABLE_GPU``.
