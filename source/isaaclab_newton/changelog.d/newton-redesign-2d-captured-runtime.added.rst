* Added ``NewtonBackend.state_version``, advanced after every solver substep, after FK, and by each asset's
  ``update(dt)``. Derived asset data recomputed on read when the state version advanced, and on every read recorded
  into a CUDA graph.
* Added :meth:`NewtonBackend.notify_model_changes <isaaclab_newton.physics.NewtonBackend.notify_model_changes>` to
  apply queued model-property edits; it ran before ``forward()``, scene-data state reads and steps.
