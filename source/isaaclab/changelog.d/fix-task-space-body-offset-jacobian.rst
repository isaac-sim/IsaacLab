Fixed
^^^^^

* Fixed the body-offset Jacobian correction in :class:`~isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction`
  and :class:`~isaaclab.envs.mdp.actions.OperationalSpaceControllerAction`. The offset is now rotated into the root
  frame by the body orientation before shifting the translational rows, and the angular rows are no longer rotated by
  the offset rotation, matching the offset frame's pose and velocity. Tasks that set ``body_offset`` (for example, the
  Franka IK tasks) now receive the Jacobian of the offset frame instead of an approximation.
