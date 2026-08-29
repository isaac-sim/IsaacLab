Changed
^^^^^^^

* **Breaking:** Replaced ``SimulationContext.services`` and ``ServiceLocator`` with
  ``SimulationContext.get_or_create_backend()``, keyed by backend type. Backend integrations should
  construct or retrieve native resources with ``sim.get_or_create_backend(BackendType, ...)`` and
  implement ``clear()`` for simulation-owned teardown; ``close()`` is no longer called by the registry.
