Changed
^^^^^^^

* Updated the supported Isaac Sim version to 6.0.1.
* Added a pre-import ``requires_kit`` test boundary so package tests can run
  without launching Isaac Sim by default.

Fixed
^^^^^

* Fixed :class:`~isaaclab.app.AppLauncher` argument filtering for the
  ``requires_kit`` pytest selector.
