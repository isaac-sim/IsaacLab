* Restored native OVRTX cloning as the default with OVPhysX, avoiding camera-task throughput regressions
  when ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` was unset. Independent OVStage rendering remained available through
  ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``, and shared physics/rendering stages through ``ISAAC_LAB_SHARE_OVSTAGE=1``.
