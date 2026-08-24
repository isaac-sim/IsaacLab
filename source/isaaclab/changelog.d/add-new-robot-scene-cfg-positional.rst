Fixed
^^^^^

* Fixed :class:`isaaclab.scene.InteractiveSceneCfg` consuming the first positional constructor
  argument as :attr:`class_type`. The field is now declared last, so scene configurations that
  pass :attr:`num_envs` and :attr:`env_spacing` positionally construct correctly again instead
  of failing validation with ``Missing values detected ... num_envs``.
* Fixed the ``add_new_robot.py`` tutorial failing at startup, and its wheel-velocity actions
  being sized for a single environment so that ``--num_envs`` greater than one raised a shape
  mismatch.
