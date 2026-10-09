Fixed
^^^^^

* Fixed the ``choice`` preset of ``IsaacContrib-Factory-Franka`` crashing on the first reset with
  ``AttributeError: 'reset_accumulator' object has no attribute 'terms'``. ``events.reset_strategies``
  in :class:`~isaaclab_tasks.contrib.nist.factory_env_cfg.FactoryEventCfg` was hardcoded to
  ``ACCUMULATOR_RESET`` for every preset, so the ``choice`` preset's curriculum callback (which reads
  ``.terms['reset_strategies']`` off a ``TermChoice``-shaped term) ran against the accumulator-shaped
  term instead. ``reset_strategies`` now switches to ``SCENE_RESET`` under the ``choice`` preset,
  matching what the curriculum callback and ``play_mode`` already expected.
