* **Breaking:** Changed ``joint_ordering="physx"`` or ``body_ordering="physx"`` on Isaac Sim PhysX to mean the
  OVPhysX order, mapped from Isaac Sim PhysX's authored order. Resolving it imports the asset with Newton and raises
  where that fails or the order is ambiguous, for example for Humanoid-AMP, UR10e with the Robotiq 2F-85 gripper and
  the gravity-compensated Rizon 4s. Checkpoints trained on Isaac Sim PhysX with ``"physx"`` used the authored order;
  replay them with explicit name lists recorded without an ordering.
