* **Breaking:** Changed ``joint_ordering="physx"`` or ``body_ordering="physx"`` on Isaac Sim PhysX to mean the
  OVPhysX order, mapped from Isaac Sim PhysX's authored order. Checkpoints trained on Isaac Sim PhysX with
  ``"physx"`` used the authored order; replay them with explicit name lists recorded without an ordering.
