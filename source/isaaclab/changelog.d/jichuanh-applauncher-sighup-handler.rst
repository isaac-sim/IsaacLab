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
  fires SIGKILL on the current process via a raw libc ``kill(2)`` syscall
  through ``ctypes`` after the deadline, from both the ``atexit`` hook
  and the abort signal handler. Used by CI to bound the upstream Kit
  shutdown hang inside ``quickReleaseFrameworkAndTerminate`` (see
  https://github.com/isaac-sim/IsaacLab/issues/3475, NVBug 5948099 /
  OMPE-75416). The hang is a GIL deadlock — Python's bytecode dispatcher
  cannot run while Kit's teardown C++ frames hold the GIL — so
  ``os._exit`` (which needs the dispatcher) cannot reliably fire from a
  daemon thread; the ctypes-issued libc syscall releases the GIL across
  the C call and is the only python-side approach that fires under the
  worst case. Off by default — interactive user code still gets the
  graceful teardown.
