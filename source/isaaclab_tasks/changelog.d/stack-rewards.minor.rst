Added
^^^^^

* Added staged reward terms for the cube stacking tasks
  (:mod:`isaaclab_tasks.contrib.stack.mdp.rewards`), built on the tasks' existing
  ``object_grasped``, ``object_stacked`` and ``cubes_stacked`` predicates so that
  reward and success cannot drift apart. The shipped task configurations are
  unchanged and still set ``rewards = None``; reinforcement-learning configurations
  opt in by assigning their own rewards group.
