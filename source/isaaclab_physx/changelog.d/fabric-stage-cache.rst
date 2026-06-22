Added
^^^^^

* Added :class:`~isaaclab_physx.sim.FabricStageCache` — a lightweight cache for the
  ``usdrt.Usd.Stage`` attachment and ``IFabricHierarchy`` handles, registered as a
  service on :class:`~isaaclab.sim.SimulationContext` via ``set_service()``.

  Multiple :class:`~isaaclab_physx.sim.views.FabricFrameView` instances now share a
  single hierarchy handle instead of each creating its own.  The cache is automatically
  closed on ``SimulationContext.clear_instance()``.
