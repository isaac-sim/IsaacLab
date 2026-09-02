Changed
^^^^^^^

* **Breaking:** Routed production PhysX cloning through the simulation-owned
  ``PhysxReplicateContext.replicate(plan)`` contract and removed ``PHYSICS_CONTEXT``, ``queue(...)``,
  and ``queue_mapping(...)``. Standalone tooling may continue to use ``physx_replicate(...)``.
