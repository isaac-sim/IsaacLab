Changed
^^^^^^^

* Changed differential IK, joint impedance, and operational space controllers to use Newton's
  model-free controller APIs.
* **Breaking:** DiffIK and OSC initialized Newton at construction and required ``cfg.num_joints``.
  Set this field for standalone callers; action terms populated it from resolved joints. Use a separate controller
  when changing the joint count; ``set_joint_pos_limits()`` remained available.
* **Breaking:** Controller solves used float32 internal buffers. Callers requiring float64
  solver precision must retain the previous implementation; output tensors remain independent snapshots.

* **Breaking:** OSC adopted Newton's motion selection before inertia decoupling and required at least
  six controlled joints for decoupling. Revalidate hybrid force/motion gains; disable
  ``inertial_dynamics_decoupling`` for under-actuated arms. Null-space posture efforts were
  mass-weighted only with decoupling enabled; retune affected posture gains.

Fixed
^^^^^

* Fixed variable joint-impedance gain clamping for robot batches with more than two joints.
