Added
^^^^^

* Added ``implementation`` to the differential IK, joint impedance, and operational-space controller
  configurations. Set ``implementation="newton"`` to run them through Newton's model-free solvers;
  the ``"isaaclab"`` implementations remain the default. The Newton differential IK and
  operational-space controllers require the new ``num_joints`` constructor argument and compute in
  float32. Newton operational-space control applies motion-axis selection before inertia
  decoupling, so hybrid force/motion tasks must revalidate their gains before opting in. It also skips the
  mass matrix in null-space posture control without inertial decoupling and ignores
  ``inertia_conditioning_thresholds``, so near-singular configurations are not damped.

Changed
^^^^^^^

* Improved :meth:`~isaaclab.controllers.OperationalSpaceController.set_command` performance by reusing a
  preallocated identity task frame and rotating the gains and selection matrices into the root frame in one
  batched step. The cost no longer grows with the number of environments.

Fixed
^^^^^

* Fixed variable joint-impedance gain clamping for robot batches with more than two joints.
* Fixed batched joint-impedance inertia compensation when the robot and joint counts differed.
* Fixed the differential IK SVD solver for position-only and under-actuated tasks.
* Fixed Isaac Lab differential IK joint-limit avoidance failing when joint limits were given as float64.
