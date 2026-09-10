Fixed
^^^^^

* Fixed a shutdown hang on Ctrl+C in ``random_agent``/``zero_agent``, every ``train``/``play``
  RL backend (``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``), and the LEAPP deployment script,
  while multiple GUI-backed visualizers were active (for example ``--visualizer newton,kit``).
  ``env.close()`` previously only ran on the happy path, so an interrupt skipped the
  visualizer teardown in :meth:`~isaaclab.sim.SimulationContext.clear_instance` and the
  process had to be killed. ``env.close()`` is now registered on a :class:`contextlib.ExitStack`
  immediately once the environment is created, guaranteeing it runs
  (``isaaclab_rl.entrypoints.common.suppressed_shutdown_guard``). ``random_agent``/
  ``zero_agent`` additionally install a flag-only ``SIGINT`` handler
  (``isaaclab_rl.entrypoints.common.flag_only_sigint``), since a raw interrupt there can land
  asynchronously inside one of Kit's own internal callbacks rather than the script's own loop.
