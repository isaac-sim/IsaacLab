Fixed
^^^^^

* Fixed ``calculate_rotation_error`` in ``source/isaaclab/test/controllers/test_pink_ik.py``
  using element-wise multiplication (``*``) instead of matrix multiplication (``@``) to
  compose two rotation matrices, producing a non-rotation matrix and propagating ``NaN``
  through ``quat_from_matrix`` (after the unit-norm guard added by
  `isaac-sim/IsaacLab#5609 <https://github.com/isaac-sim/IsaacLab/pull/5609>`_). The latent
  bug was introduced in `isaac-sim/IsaacLab#3149
  <https://github.com/isaac-sim/IsaacLab/pull/3149>`_ and masked for ~9 months because the
  Hadamard and matrix products of two near-identity rotation matrices are close enough that
  ``quat_from_matrix`` could still return a near-unit quaternion. Once IK no longer
  converged to literal identity (e.g., G1 envs or any seed perturbation), the assertion
  ``Left hand IK rotation error (nan) exceeds tolerance`` started firing.
