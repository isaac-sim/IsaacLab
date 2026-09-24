Fixed
^^^^^

* Restored legacy OVRTX rendering throughput with the pinned OvPhysX runtime set on Linux x86_64 by initializing
  OVStage's GPU hierarchy before OVRTX cached its cuBRIC adapter. Set ``ISAAC_LAB_OVRTX_PREWARM_CUBRIC=0`` to
  disable the temporary workaround.
