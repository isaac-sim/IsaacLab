Fixed
^^^^^

* Fixed :class:`~isaaclab.app.AppLauncher` registering a Python handler for
  ``SIGABRT``. fatal signals stay at ``SIG_DFL`` so crashes terminate and can
  produce core dumps; graceful shutdown remains on ``SIGTERM`` only.
