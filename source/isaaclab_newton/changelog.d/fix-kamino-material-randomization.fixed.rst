* Fixed :class:`~isaaclab_newton.envs.mdp.randomize_rigid_body_material` sharing sampled materials across all
  environments for Kamino with Newton's collision pipeline, the default. Only Kamino's own collision detector
  shares material groups; with Newton's pipeline each environment and shape is sampled as for the other solvers.
