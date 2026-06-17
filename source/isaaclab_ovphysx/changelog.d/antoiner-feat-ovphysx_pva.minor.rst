Added
^^^^^

* Added :class:`~isaaclab_ovphysx.sensors.Pva` and
  :class:`~isaaclab_ovphysx.sensors.PvaData` implementing the
  :class:`~isaaclab.sensors.pva.BasePva` /
  :class:`~isaaclab.sensors.pva.BasePvaData` contracts on the OVPhysX
  backend. Reports world-frame pose, body-frame linear and angular
  velocities, body-frame coordinate linear and angular accelerations,
  and projected gravity using ovphysx tensor bindings on the rigid-body
  ancestor of the sensor prim path. Linear and angular accelerations
  are coordinate accelerations (zero at rest, ``-g`` in freefall) and
  do not include the IMU's gravity bias — projected gravity is reported
  separately as the unit gravity direction vector.
* Added :meth:`~isaaclab_ovphysx.physics.OvPhysxManager.get_gravity`
  classmethod mirroring PhysX's ``SimulationView.get_gravity()`` so
  backend-agnostic sensor code can read scene gravity through a single
  entry point.
