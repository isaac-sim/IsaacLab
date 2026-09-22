Fixed
^^^^^

* Fixed core Franka tasks to use the canonical flat asset with explicit backend-specific physics
  payloads and fast gripper-only collisions by default. Full primitive arm collisions are available
  through the ``arm_collisions`` domain preset.
* Fixed Lift and Reorient reset-clearance sampling to ignore disabled collision schemas.
* Fixed automatic PhysX Lift selection to use the homogeneous object setup supported by the
  kitless OvPhysX fast replication path.
* Fixed absolute differential-IK Reach actions to cover the configured Cartesian command workspace.
* Fixed Franka operational-space control to retain the asset's solver limits and joint properties.
* Restored continuous Reach tracking by removing the early success termination and restoring the
  fine-grained position reward, while retaining position-and-orientation success metrics.
* Restored symmetric point-cloud noise in the Lift and Reorient ADR curriculum.
* Restored action-rate and joint-velocity regularization for rigid Lift and Reorient training.
* Kept terminal non-finite rigid-object states from propagating into Lift and Reorient reward batches.
* Made Franka Reorient train from scratch by starting its ADR goal curriculum from stable, shape-aware
  pre-grasps with a 1 mm contact preload; rigid Lift behavior was unchanged.
