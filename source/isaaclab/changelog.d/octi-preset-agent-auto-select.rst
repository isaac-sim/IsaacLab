Fixed
^^^^^

* Fixed ``isaaclab benchmark training`` and ``isaaclab benchmark play`` for skrl
  not resolving the agent entry point from the task's registry metadata. Both
  ``_parse_args`` functions now pass ``agent_library="skrl"`` to
  :func:`~isaaclab_tasks.utils.setup_preset_cli`, so a benchmark run picks the
  same agent config that ``isaaclab train`` does for the active preset.
* Fixed ``isaaclab benchmark play`` reporting the raw ``--algorithm`` flag in its
  KPI metadata while writing the resolved algorithm to the run manifest. Both now
  report the resolved value, matching ``isaaclab benchmark training``.
