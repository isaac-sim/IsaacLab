Changed
^^^^^^^

* **Breaking:** Routed production Newton cloning through the simulation-owned
  ``NewtonReplicateContext.replicate(plan)`` contract and removed ``PHYSICS_CONTEXT`` and
  ``queue_mapping(...)``. Standalone tooling may continue to use ``newton_physics_replicate(...)``
  with NumPy arrays; its unused ``device`` argument was removed. Changed world-builder hooks to
  receive independent NumPy arrays for the environment position and orientation instead of Python lists.
