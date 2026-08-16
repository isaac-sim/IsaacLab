Fixed
^^^^^

* Fixed the ``add_new_robot.py`` tutorial mutating the Dofbot's stored default joint positions through a
  zero-copy Torch view while constructing its wave command.
