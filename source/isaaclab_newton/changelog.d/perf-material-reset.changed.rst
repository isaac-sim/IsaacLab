* Reused device-resident material sampling bounds and shape indices in Newton material
  randomization to avoid repeated host-to-device copies and CUDA synchronization on reset.
