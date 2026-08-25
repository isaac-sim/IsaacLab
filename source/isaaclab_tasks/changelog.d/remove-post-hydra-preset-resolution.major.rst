Changed
^^^^^^^

* **Breaking:** Made task composition the sole owner of preset replacement. Runtime task code now
  requires concrete camera, renderer, and physics configurations; use
  :func:`isaaclab_tasks.utils.resolve_task_config` or :func:`isaaclab_tasks.utils.parse_env_cfg`
  instead of passing a raw registered configuration class to an environment.
* Added explicit programmatic ``overrides`` to :func:`isaaclab_tasks.utils.resolve_task_config` so
  tools and tests can use the same composition path without modifying :data:`sys.argv`.
