Fixed
^^^^^

* Fixed :func:`~isaaclab.app.launch_simulation` passing a successful exit status to runtime launchers
  after an unhandled Ctrl+C; it now passed exit status 130.
