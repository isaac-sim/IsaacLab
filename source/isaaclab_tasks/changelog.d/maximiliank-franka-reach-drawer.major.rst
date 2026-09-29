Changed
^^^^^^^

* Migrated maintained Franka Reach, Drawer, and related contributed tasks to the shared flat asset with
  backend-specific physics and gripper-only collisions by default. Select ``arm_collisions`` where arm
  contacts are required. Existing checkpoints should be requalified against the changed robot dynamics.

Fixed
^^^^^

* Corrected the Reach action and controller contracts for the shared asset.
