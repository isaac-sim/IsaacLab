* Changed ``FrankaReachEnvCfg`` and ``AnymalCFlatEnvCfg`` to hold ``sim.dt``, ``sim.render_interval`` and
  ``decimation`` as preset alternatives for ``newton_nerd``. ``AnymalCFlatEnvCfg`` also holds its leg actuators,
  joint-target clipping and startup randomization events as alternatives. Resolve the configuration with
  :func:`~isaaclab_tasks.utils.hydra.resolve_task_config` or :func:`~isaaclab_tasks.utils.hydra.resolve_presets`
  before reading these fields.
* Changed ``Isaac-Reach-Franka-OSC`` to derive from the new ``FrankaReachBaseEnvCfg``, which keeps the neural
  preset out of the operational space task, and the ANYmal C navigation task to take its low-level configuration
  with the default presets resolved.
