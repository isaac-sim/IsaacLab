Changed
^^^^^^^

* **Breaking:** Replaced ``SimulationContext.services`` and ``ServiceLocator`` with
  ``SimulationContext.get_or_create_backend()``, keyed by backend type and the plain ``resource_key``
  carried by physics, renderer, and visualizer configs. Backend integrations should construct or retrieve
  native resources with ``sim.get_or_create_backend(BackendType, ..., resource_key=cfg.resource_key)``
  and implement ``clear()`` for simulation-owned teardown; ``close()`` is no longer called by the registry.
