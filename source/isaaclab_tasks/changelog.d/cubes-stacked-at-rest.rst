Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.contrib.stack.mdp.terminations.cubes_stacked` reporting success for a
  cube that is still falling. The check tested an instantaneous configuration, which a cube released
  above its target satisfies on the way down, so a dropped cube was scored as a completed stack and
  Mimic wrote the episode into the generated dataset as a demonstration of the task. The cubes must
  now also be at rest, controlled by the new ``max_lin_vel`` argument (default ``0.05`` m/s,
  ``None`` to skip the check).

  This slightly lowers reported success rates for the stack tasks, both in Mimic data generation and
  in policy evaluation, because episodes that end with the cube on the table no longer count. Pass
  ``max_lin_vel=None`` in the termination term's ``params`` to restore the previous behaviour.
