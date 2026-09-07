Fixed
^^^^^

* Fixed the ``rsl_rl``, ``rl_games`` and ``sb3`` benchmark train and play entrypoints not passing
  ``agent_library`` to :func:`~isaaclab_tasks.utils.setup_preset_cli`, which disabled preset-based
  ``--agent`` selection and the registered-agent help listing for those backends. Only the ``skrl``
  entrypoints wired it.
