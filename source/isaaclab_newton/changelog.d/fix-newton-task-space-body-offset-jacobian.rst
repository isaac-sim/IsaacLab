Fixed
^^^^^

* Fixed the body-offset Jacobian of
  :class:`~isaaclab_newton.envs.mdp.actions.NewtonDifferentialInverseKinematicsAction` and
  :class:`~isaaclab_newton.envs.mdp.actions.NewtonOperationalSpaceControllerAction`. The offset is now rotated into
  the root frame by the body orientation before shifting the translational rows, and the angular rows are no longer
  rotated by the offset rotation, matching the target frame's pose and velocity.
