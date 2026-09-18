Fixed
^^^^^

* Fixed :class:`~isaaclab.app.AppLauncher` registering a Python handler for
  ``SIGABRT``. fatal-signal dispositions are left untouched (default or
  carb's crash handler); graceful shutdown remains on ``SIGTERM`` only.
