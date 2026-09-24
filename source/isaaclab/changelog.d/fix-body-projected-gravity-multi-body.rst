Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.mdp.observations.body_projected_gravity_b` raising an error when more than one body is
  selected, including the default all-body selection. Gravity is now projected into every selected body frame.
