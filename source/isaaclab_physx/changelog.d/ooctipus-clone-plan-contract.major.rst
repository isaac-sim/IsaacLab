Changed
^^^^^^^

* **Breaking:** Routed PhysX cloning through the simulation-owned
  ``PhysxReplicateContext.replicate(plan)`` contract. Replace ``physx_replicate(...)``,
  ``PHYSICS_CONTEXT``, ``queue(...)``, and ``queue_mapping(...)`` calls with a registered
  ``PhysxReplicateContext`` receiving the shared :class:`isaaclab.cloner.ClonePlan`.
