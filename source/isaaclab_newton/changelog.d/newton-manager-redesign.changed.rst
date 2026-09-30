* **Breaking:** Rewrote :class:`~isaaclab_newton.physics.NewtonManager` as a facade over construction requests and a
  per-model runtime. Everything bound to a finalized model (solver, contacts, sensors, actuators, consumer stages,
  and captured graphs) now lives on one runtime that a hard reset discards, replacing about 60 class attributes.
  Code that read private manager attributes must use the accessors listed under Added.
* **Breaking:** Replaced the separate full and physics-only stepping paths with one compiled step program. Newton
  actuators that are not CUDA-graph-safe now run eagerly between captured segments instead of forcing the
  environment to own the decimation loop, so :meth:`~isaaclab_newton.physics.NewtonManager.handles_decimation`
  reports whether Newton actuators are active, independent of their graph safety.
* **Breaking:** :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor`,
  :meth:`~isaaclab_newton.physics.NewtonManager.add_frame_transform_sensor`, and
  :meth:`~isaaclab_newton.physics.NewtonManager.add_imu_sensor` return the Newton sensor instead of a key or index.
* **Breaking:** ``NewtonManager.transforms_may_change_on_graph_replay`` is now a method.
* Double-buffered solvers alternate state buffers when the program is compiled instead of swapping them in Python,
  so :attr:`~isaaclab_newton.physics.NewtonBackend.state_0` stays the same object across steps.
