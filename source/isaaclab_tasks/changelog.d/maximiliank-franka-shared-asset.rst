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
* Restored continuous Franka Reach tracking by removing the early success termination and restoring the
  fine-grained position reward, while retaining position-and-orientation success metrics.
* Restored symmetric point-cloud noise in the Lift and Reorient ADR curriculum.
* Restored success-driven action-rate and joint-velocity regularization for rigid Lift and Reorient
  training, avoiding a fixed-step penalty jump before grasping is learned.
* Kept Franka Lift and Reorient's abnormal-state termination from penalizing ordinary exploration
  at the actuator's nominal velocity limits.
* Kept terminal non-finite rigid-object states from propagating into Lift and Reorient reward batches.
* Retained aligned Franka Lift pre-grasps in the reset bank and applied their finger opening after the
  generic gripper reset. Re-evaluate existing Lift checkpoints because the training reset distribution changed.
