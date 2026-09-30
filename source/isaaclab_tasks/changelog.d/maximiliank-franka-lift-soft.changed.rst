* Migrated Franka rigid and deformable Lift tasks to the shared flat asset, with backend-specific physics
  and gripper-only collisions by default. Select ``arm_collisions`` to enable primitive arm contacts.
  Default reset clearance checks now cover the hand and fingers only; ``arm_collisions`` also checks the arm.
