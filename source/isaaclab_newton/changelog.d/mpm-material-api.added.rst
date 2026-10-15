* Added MPM particle material getters and setters with validated parameter updates and density-consistent particle masses.
  Preserved zero-density kinematic particles on write-back, avoided host index allocation for
  regular selections, and rejected repeated writes while snapshotting aliased material inputs.
