* Fixed :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` compounding its samples across
  calls. It now randomizes from the tendon properties as they were before its first write; with
  ``operation="scale"`` the tendon stiffness and damping previously drifted multiplicatively on every reset, which
  drove Shadow Hand training with the ``randomized`` preset to non-finite observations.
