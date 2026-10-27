* Changed ``teleop_se3_agent.py`` and ``record_demos.py`` to no longer force Kit, so keyboard, gamepad
  and SpaceMouse teleoperation also run with kitless visualizers such as ``newton_rtx``. ``--xr`` and
  tasks that need Kit still start it.
