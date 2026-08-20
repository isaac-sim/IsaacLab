Fixed
^^^^^

* Fixed :class:`~isaaclab.markers.VisualizationMarkers` creating an unpumped Kit/USD marker
  backend for non-Kit-pumping visualizers (e.g. ``newton_gl``). The Kit backend's raw USD marker
  writes were never digested by Fabric without a Kit ``app.update()`` pump, desyncing the
  point-instancer prototype table (``FabricManager::initializePointInstancer mismatched
  prototypes``) and crashing the next PhysX GPU articulation step with ``CUDA error: unspecified
  launch failure``. Backend selection now checks for an active GUI, RTX sensor rendering, XR, or
  offscreen capture directly instead of the broader :attr:`~isaaclab.sim.SimulationContext.is_rendering`,
  which is also true for visualizers that never pump Kit.
