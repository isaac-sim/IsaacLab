Removed
^^^^^^^

* **Breaking:** Removed late task-preset resolution from environment construction and the
  :func:`isaaclab.utils.resolve_cfg_presets` helper. Compose registered tasks with
  :func:`isaaclab_tasks.utils.resolve_task_config` or :func:`isaaclab_tasks.utils.parse_env_cfg`
  before constructing an environment.
* **Breaking:** Replaced ``run_config_from_presets`` with ``run_config_from_env_cfg`` in benchmark
  capture. Pass the concrete composed environment configuration instead of inferring backends from
  selector strings.
