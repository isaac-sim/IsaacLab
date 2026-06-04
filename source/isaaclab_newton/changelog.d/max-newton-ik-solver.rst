Added
^^^^^

* Added :class:`~isaaclab_newton.ik.NewtonIKSolver` and
  :class:`~isaaclab_newton.envs.mdp.actions.NewtonInverseKinematicsAction`
  for Newton-backed inverse kinematics, including named pose objectives and
  custom Newton objective passthrough.
* Added persistent IK seeds and helpers to initialize pose-objective targets
  from live Newton body transforms.
