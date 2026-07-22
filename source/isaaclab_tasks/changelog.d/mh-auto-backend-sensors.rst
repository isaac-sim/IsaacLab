Changed
^^^^^^^

* Changed the rough-terrain velocity tasks to declare their height scanner and contact sensor with
  the backend-dispatching :class:`~isaaclab.sensors.RayCasterCfg` and
  :class:`~isaaclab.sensors.ContactSensorCfg`, which select the matching backend implementation
  automatically. This removes the per-task backend sensor presets.
