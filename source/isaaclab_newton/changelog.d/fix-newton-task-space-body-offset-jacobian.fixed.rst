* Fixed the body-offset Jacobian of
  :class:`~isaaclab_newton.envs.mdp.actions.NewtonDifferentialInverseKinematicsAction` and
  :class:`~isaaclab_newton.envs.mdp.actions.NewtonOperationalSpaceControllerAction`. The translational rows are now
  shifted by the offset rotated with the body orientation, and the angular rows are no longer rotated by the offset
  rotation, matching the target frame's pose and velocity.
