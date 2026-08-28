Fixed
^^^^^

* Fixed the ``isaaclab teleop run``, ``record``, and ``replay`` workflows rejecting Hydra-style
  task selectors such as ``physics=isaacsim_physx presets=diffik``. The workflows now resolve task
  configurations through the shared preset-aware path and expose the selector syntax in ``--help``.
