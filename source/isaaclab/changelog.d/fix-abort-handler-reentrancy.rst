Fixed
^^^^^

* Fixed the abort-signal handler of :class:`~isaaclab.app.AppLauncher` re-entering
  ``SimulationApp.close()`` when a signal arrived while the app was already closing.
  The nested calls recursed until the stack overflowed, which prevented distributed
  training workers from shutting down on ``SIGTERM`` (the launcher had to escalate to
  ``SIGKILL``) and masked the original failure signal behind a spurious segmentation
  fault. A re-entrant signal now falls back to the default signal action so the
  process terminates with the original signal.
