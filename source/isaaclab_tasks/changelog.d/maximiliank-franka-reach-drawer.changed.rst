* **Breaking:** Migrated maintained Franka Reach, Drawer, and related contributed tasks to the main
  ``FRANKA_PANDA_CFG`` shared flat asset with backend-specific physics and gripper-only collisions by default.
  Select ``arm_collisions`` where arm contacts are required. Existing checkpoints should be requalified against the changed robot dynamics.
* **Breaking:** Changed Franka Reach and Reach-OSC to continuous pose tracking: success remained a reported metric,
  but no longer ended the episode or awarded the terminal success bonus. Episodes ran until timeout,
  matching the qualified tracking checkpoints; reward totals changed accordingly.
