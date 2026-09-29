Fixed
^^^^^

* Fixed the ``scale`` operation of :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_mass`,
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_inertia`,
  :class:`~isaaclab.envs.mdp.events.randomize_actuator_gains`,
  :class:`~isaaclab.envs.mdp.events.randomize_joint_parameters` and
  :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` rejecting valid
  ``distribution="gaussian"`` parameters. They are now validated as ``(mean, std)`` instead of a
  ``(low, high)`` range.
