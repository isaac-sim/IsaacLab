Added
^^^^^

* Added an ``overrides`` argument to :func:`~isaaclab_tasks.utils.parse_env_cfg` for applying Hydra-style
  ``key=value`` overrides (e.g. ``physics=isaacsim_physx``) to a task's registered configuration from standalone
  scripts that do not use the full Hydra CLI. The ``run_cartpole_rl_env.py`` tutorial now forwards unrecognized
  command-line arguments this way, which is required to select the PhysX backend for the OVD Recorder.
