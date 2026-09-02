Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.contrib.stack.mdp.terminations.cubes_stacked` reporting success for a
  cube that is still falling: the check tested an instantaneous configuration, which a dropped cube
  satisfies on the way down. Cubes must now also be at rest, controlled by the new ``max_lin_vel``
  argument (default ``0.05`` m/s). Reported stack success rates drop slightly as a result; pass
  ``max_lin_vel=None`` to restore the previous behaviour.
