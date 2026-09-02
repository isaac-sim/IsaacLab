Changed
^^^^^^^

* **Breaking:** Routed Newton cloning through the simulation-owned
  ``NewtonReplicateContext.replicate(plan)`` contract. Replace ``newton_physics_replicate(...)``,
  ``PHYSICS_CONTEXT``, and ``queue_mapping(...)`` calls with a registered
  ``NewtonReplicateContext`` receiving the shared :class:`isaaclab.cloner.ClonePlan`.
