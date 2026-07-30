Fixed
^^^^^

* Fixed :meth:`~isaaclab.controllers.DifferentialIKController.set_command` producing a NaN target
  orientation when an absolute pose command carried a zero-norm quaternion. The NaN propagated into
  the joint position targets and diverged the articulation, surfacing a step later as an unrelated
  ``torch.linalg.solve ... input matrix is singular`` error. Degenerate quaternions now hold the
  current end-effector orientation instead.
* Fixed the ``adaptive_dls`` inverse-kinematics method reporting a non-finite Jacobian as an opaque
  LAPACK singular-matrix or convergence failure. It now raises an error naming the actual cause: the
  articulation state diverged before the solve.
