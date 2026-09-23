Added
^^^^^

* Added ``implementation`` to the differential IK, joint impedance, and operational-space controller
  configurations. Set ``implementation="newton"`` to run them through Newton's model-free solvers; the
  ``"native"`` implementations remain the default. Newton computes in float32, and Newton
  differential IK with joint-limit avoidance requires
  :meth:`~isaaclab.controllers.DifferentialIKController.set_joint_pos_limits` before the first
  ``compute()``. Newton operational-space control applies motion-axis selection before inertia
  decoupling, so hybrid force/motion tasks must revalidate their gains before opting in.

Fixed
^^^^^

* Fixed variable joint-impedance gain clamping for robot batches with more than two joints.
* Fixed batched joint-impedance inertia compensation when the robot and joint counts differed.
* Fixed the differential IK SVD solver for position-only and under-actuated tasks.
