Fixed
^^^^^

* Fixed :attr:`~isaaclab.app.AppLauncher.has_window` being accidentally removed in
  `#6658 <https://github.com/isaac-sim/IsaacLab/pull/6658>`_, which caused an
  ``AttributeError`` in ``teleop_se3_agent.py`` and ``record_demos.py`` on every
  IsaacTeleop run.
* Fixed XR sessions without an explicit windowed visualizer no longer forcing headless
  mode, caused by the same PR renaming ``_xr_implies_headless`` to ``_xr_auto_start``
  without updating the enforcement block in ``_resolve_headless_settings``.
