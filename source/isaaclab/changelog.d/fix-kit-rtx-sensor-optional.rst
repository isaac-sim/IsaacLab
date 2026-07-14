Fixed
^^^^^

* Fixed the ``isaaclab.python.kit`` experience failing to start on Isaac Sim
  builds that do not ship ``isaacsim.sensors.experimental.rtx`` by marking the
  extension as optional, matching the handling of other experimental extensions.
