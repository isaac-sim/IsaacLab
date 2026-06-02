Fixed
^^^^^

* Handled ``SIGHUP`` in :class:`~isaaclab.app.AppLauncher` so the
  simulation app shuts down cleanly when the controlling session leader
  exits (e.g. parent shell supervising sibling shards in multi-GPU CI).
  Previously SIGHUP terminated the process with default disposition,
  bypassing :meth:`SimulationApp.close` and leaving USD/PhysX state
  attached for the next sibling shard ("Stage X already attached" log
  line and downstream shutdown hangs).
* Made :meth:`~isaaclab.app.AppLauncher._abort_signal_handle_callback`
  exit the process after closing the app. The previous implementation
  swallowed the signal's terminate semantics, allowing Python to
  resume past a SIGTERM/SIGABRT/SIGSEGV and leaving Kit half-torn-down.

Added
^^^^^

* Added ``ISAACLAB_FORCE_EXIT_TIMEOUT`` env var (integer seconds) for
  :class:`~isaaclab.app.AppLauncher`. When set, arms a daemon thread that
  calls ``os._exit(0)`` after the deadline, from both the ``atexit`` hook
  and the abort signal handler. Used by CI to bound the upstream Kit
  shutdown hang at
  ``shutdown_and_release_framework`` (see
  https://github.com/isaac-sim/IsaacLab/issues/3475). The hang sits
  inside Kit binary code; ``skip_cleanup=True`` enters the same code
  path and Python signal handlers cannot interrupt it, so ``os._exit``
  is the only python-side escape. Off by default — interactive user
  code still gets the graceful teardown.
