* Fixed :class:`~isaaclab.envs.mdp.events.reset_joints_within_limits_range` clamping every joint with the soft
  limits of the listed joints, which moved unlisted joints off their defaults, failed for joint subsets and applied
  the wrong limits for out-of-order keys. Empty range dictionaries no longer raise an ``IndexError``.
