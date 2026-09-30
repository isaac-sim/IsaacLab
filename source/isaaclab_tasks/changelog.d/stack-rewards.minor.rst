Added
^^^^^

* Added staged reward terms for the cube stacking tasks
  (:mod:`isaaclab_tasks.contrib.stack.mdp.rewards`), built on the tasks' existing
  ``object_grasped``, ``object_stacked`` and ``cubes_stacked`` predicates so that
  reward and success cannot drift apart. The shipped task configurations are
  unchanged and still set ``rewards = None``; reinforcement-learning configurations
  opt in by assigning their own rewards group.

* Added :class:`FrankaStackRuntimeDRCfg`, a cube-stacking configuration with
  runtime visual domain randomization attached. It lives here rather than beside
  the runtime because it is a task configuration and imports ``isaaclab_tasks``;
  ``isaaclab_contrib`` cannot depend on ``isaaclab_tasks``, since the dependency
  runs the other way through ``isaaclab_assets``.

* Added a Franka stacking configuration and demo scripts for runtime visual domain
  randomization, with segmentation controls and preserved robot/table/object masks.

Changed
^^^^^^^

* Changed the runtime DR stacking task to inherit zero boundary dilation and
  disabled foreground compositing. Set ``boundary_px=1`` and
  ``composite_foreground=True`` explicitly to restore its previous behavior.
