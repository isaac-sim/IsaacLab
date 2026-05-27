Fixed
^^^^^

* Fixed translational Jacobian drift in
  :meth:`~isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction._compute_frame_jacobian`
  when called multiple times per step with non-``None`` ``body_offset``.
