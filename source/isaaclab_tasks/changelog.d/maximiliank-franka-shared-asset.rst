Fixed
^^^^^

* Fixed core Franka tasks to use the shared Menagerie asset with backend-specific physics payloads
  and a fast gripper-only collision preset for their end-effector contact scope, while preserving
  the source model's passive joint damping across backends.
* Fixed automatic PhysX Lift selection to use the homogeneous object setup supported by the
  kitless OvPhysX fast replication path.
* Fixed absolute differential-IK Reach actions to cover the configured Cartesian command workspace.
* Fixed Franka operational-space control to retain the asset's solver limits and joint properties.
* Restored symmetric point-cloud noise in the Lift and Reorient ADR curriculum.
* Restored action-rate and joint-velocity regularization for rigid Lift and Reorient training.
* Kept terminal non-finite rigid-object states from propagating into Lift and Reorient reward batches.
