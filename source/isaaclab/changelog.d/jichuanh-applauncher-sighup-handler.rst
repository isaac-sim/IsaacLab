Fixed
^^^^^

* Handled ``SIGHUP`` in :class:`~isaaclab.app.AppLauncher` so the
  simulation app shuts down cleanly when the controlling session leader
  exits (e.g. parent shell supervising sibling shards in multi-GPU CI).
  Previously SIGHUP terminated the process with default disposition,
  bypassing :meth:`SimulationApp.close` and leaving USD/PhysX state
  attached for the next sibling shard ("Stage X already attached"
  log line and downstream shutdown hangs).
* Made :meth:`~isaaclab.app.AppLauncher._abort_signal_handle_callback`
  exit the process after closing the app. The previous implementation
  swallowed the signal's terminate semantics, allowing Python to
  resume past a SIGTERM/SIGABRT/SIGSEGV and leaving Kit half-torn-down.
