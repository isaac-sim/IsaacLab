Changed
^^^^^^^

* Changed differential IK, joint impedance, and operational space controllers to use Newton's
  model-free controller APIs. Updated the Newton dependency to 1.6.0.
* **Breaking:** DiffIK and OSC required ``num_joints=`` at construction and kept their topology
  fixed. Pass the selected joint count; provide initial ``joint_pos_limits=`` for DiffIK
  joint-limit avoidance. Update custom subclasses to forward these constructor arguments.
* **Breaking:** Controller solves used float32 internal buffers. Callers requiring float64
  solver precision must retain the previous implementation; output tensors remain independent snapshots.

* **Breaking:** OSC adopted Newton's motion selection before inertia decoupling and required at least
  six controlled joints for decoupling. Revalidate hybrid force/motion gains; disable
  ``inertial_dynamics_decoupling`` for under-actuated arms. Null-space posture efforts were
  mass-weighted only with decoupling enabled; retune affected posture gains.

Fixed
^^^^^

* Fixed variable joint-impedance gain clamping for robot batches with more than two joints.
