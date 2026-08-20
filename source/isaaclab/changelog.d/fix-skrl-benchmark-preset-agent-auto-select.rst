Fixed
^^^^^

* ``isaaclab benchmark training`` and ``isaaclab benchmark play`` for skrl now
  automatically resolve the correct agent entry point when a preset is active,
  matching the behaviour of ``isaaclab train``. The ``_parse_args`` functions in
  :mod:`~isaaclab.benchmark.entrypoints.backends.skrl` now pass
  ``agent_library="skrl"`` to :func:`~isaaclab_tasks.utils.setup_preset_cli`.
