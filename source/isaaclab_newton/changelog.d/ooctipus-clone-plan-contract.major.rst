Changed
^^^^^^^

* **Breaking:** Routed production Newton cloning through the simulation-owned
  ``NewtonReplicateContext.replicate(plan)`` contract and removed ``PHYSICS_CONTEXT`` and
  ``queue_mapping(...)``. Standalone tooling may continue to use ``newton_physics_replicate(...)``.
