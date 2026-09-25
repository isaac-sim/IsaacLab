Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.events.randomize_joint_parameters` choosing the friction write path by the
  physics manager name. It now randomizes dynamic friction whenever the asset exposes it, which also covers
  OVPhysX explicitly, and its invalid-operation error names the right term.
* Fixed :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_com` choosing the center of mass layout by the
  physics manager name; it passes poses to every backend.
* Fixed :func:`~isaaclab.envs.mdp.events.randomize_rigid_body_scale` detecting articulations by class name
  instead of :class:`~isaaclab.assets.BaseArticulation`.
