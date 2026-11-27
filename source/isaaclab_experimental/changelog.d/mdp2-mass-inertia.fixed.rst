* Fixed :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_mass` scaling the inertias from before the
  startup events: an inertia offset from :data:`~isaaclab_experimental.mdp2.mdp.randomize_rigid_body_inertia` that
  ran before it was lost. It now scales the inertias at its first call, as MDP1 does.
