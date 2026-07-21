Fixed
^^^^^

* Fixed the abort-signal handler of :class:`~isaaclab.app.AppLauncher` re-entering
  ``SimulationApp.close()`` when a signal arrived while the app was already closing.
  The nested calls recursed until the stack overflowed, which prevented distributed
  training workers from shutting down on ``SIGTERM`` (the launcher had to escalate to
  ``SIGKILL``) and masked the failure behind a spurious segmentation fault. A
  re-entrant signal now falls back to the default signal action. The ``atexit`` close
  arms the same guard so a signal during a normal shutdown cannot start a nested
  teardown.
* Fixed signal-terminated processes reporting a successful exit status. Kit fast
  shutdown terminated the process with exit code 0 from inside the graceful close, so
  a ``SIGTERM``-ed worker was recorded as succeeded by distributed launchers. The
  handler now closes the app with the conventional killed-by-signal status
  (``close(exit_code=128 + signum)``; the app performs its full teardown before
  exiting on ``isaacsim.simulation_app`` >= 2.18.5) and re-raises the signal with the
  default action if the close returns.
* Fixed ``Ctrl-C`` terminating the process with exit code 0 before ``finally`` blocks
  or ``KeyboardInterrupt`` handlers in user code could run. ``SIGINT`` is restored to
  Python's default handler; the app is still closed by the ``atexit`` callback.
* Fixed the ``atexit`` close replacing the exit status of an unhandled exception with
  a successful exit code 0 under Kit fast shutdown. The close now passes a failure
  exit code when an exception is pending.

Changed
^^^^^^^

* Changed :class:`~isaaclab.app.AppLauncher` to no longer intercept ``SIGSEGV`` at the
  Python level. A Python handler cannot run for a synchronous segfault on the main
  thread, reported a successful exit for segfaults on worker threads, and replaced the
  carb crash reporter's handler. Native crashes now follow the default signal action,
  so the crash reporter can produce minidumps again.
