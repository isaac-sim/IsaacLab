Fixed
^^^^^

* Stopped :class:`~isaaclab.app.AppLauncher` from intercepting ``SIGABRT``, so aborts keep the default
  action, core dumps, and the carb crash reporter, as ``SIGSEGV`` already did.

Changed
^^^^^^^

* Replaced the private ``AppLauncher._SimulationAppLifecycle`` class with
  :func:`isaaclab_physx.app.install_exit_handlers`. Exit statuses are unchanged: ``SIGTERM`` exits
  with ``128 + signum``, an unhandled exception exits with 1, and ``SIGINT`` raises
  :class:`KeyboardInterrupt`. ``AppLauncher`` no longer prints the Kit version, kernel, and git hash
  at startup; Kit's own log still reports them.
