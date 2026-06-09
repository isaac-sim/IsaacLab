Fixed
^^^^^

* Fixed the ``isaaclab.python.kit`` GUI experience failing to start with a Kit
  dependency-solver error on Isaac Sim builds that do not ship
  ``isaacsim.robot.experimental.wheeled_robots`` or
  ``isaacsim.robot.wheeled_robots.nodes``. These extensions are not imported by
  Isaac Lab and are now declared optional, so the experience loads regardless of
  the Isaac Sim build.
