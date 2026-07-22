Changed
^^^^^^^

* Changed rough-terrain velocity tasks to use
  :class:`~isaaclab_newton.sensors.NewtonRaycastSensor` for height scans with
  Newton physics. PhysX and OvPhysX presets continue to use
  :class:`~isaaclab.sensors.RayCaster`. No task configuration changes are required.
