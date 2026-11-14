* Added GPU Fabric particle-point updates on Kit 110.4 and newer. Automatic selection
  retained the procedural GPU path and compatibility fallbacks. Set
  :attr:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg.points_update_mode`
  to ``"gpu"`` to enable streamed GPU Points on a verified compatible runtime, or
  ``"cpu"`` to retain host transport. GPU particle storage does not remove all native
  rendering or solver synchronization.
