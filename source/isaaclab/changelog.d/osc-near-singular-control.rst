Fixed
^^^^^

* Fixed excessive operational-space controller efforts near kinematic singularities by selectively damping
  poorly conditioned task-inertia directions and using the same damping for full-inertia posture control.
  Added ``inertia_conditioning_thresholds`` to configure this transition. Actuator effort limits still
  need to be enforced separately.
  Avoided explicit inertia inverses and full-inertia posture projectors, and skipped eigendecomposition
  for task inertias whose conditioning was certified by a shifted Cholesky factorization.
