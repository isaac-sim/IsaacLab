* Changed the shared Reach, locomotion velocity and Allegro physics configurations to declare the neural presets
  next to the numerical solvers. Only tasks that set a model checkpoint can run them; the others reject the preset
  with a message that the task declares no model.
* Changed ``ReachEnvCfg`` and ``AnymalCFlatEnvCfg`` to hold ``sim.dt``, ``sim.render_interval`` and ``decimation``
  as preset alternatives for ``newton_nerd``. ``AnymalCFlatEnvCfg`` also holds its leg actuators, joint-target
  clipping and startup randomization events as alternatives. Resolve the configuration with
  :func:`~isaaclab_tasks.utils.hydra.resolve_task_config` or :func:`~isaaclab_tasks.utils.hydra.resolve_presets`
  before reading these fields. The ANYmal C navigation task takes its low-level configuration with the default
  presets resolved.
