Added
^^^^^

* Added :func:`isaaclab_physx.app.install_exit_handlers`, the exit-status policy for a process that
  owns a Kit :class:`SimulationApp`: truthful exit codes on normal exit and ``SIGTERM``, default
  ``SIGINT``, and untouched fatal signals.
