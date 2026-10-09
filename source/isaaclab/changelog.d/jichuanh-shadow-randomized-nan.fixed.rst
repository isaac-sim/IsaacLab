* Fixed :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` drifting tendon values across resets.
  It now randomizes from the values read when the term is created and ignores values written afterwards by other code.
