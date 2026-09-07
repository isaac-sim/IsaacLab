Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonManager` to upload the sensor-graph task flags only
  when they change, avoiding a synchronizing pageable host-to-device copy on every camera render.
