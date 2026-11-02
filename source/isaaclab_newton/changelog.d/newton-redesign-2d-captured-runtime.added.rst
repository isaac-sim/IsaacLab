* Added ``NewtonBackend.live``, true while a step runs or a CUDA graph is recorded. Derived asset data recomputed on
  read in that interval instead of using its timestamp cache.
* Added :meth:`NewtonBackend.notify_model_changes <isaaclab_newton.physics.NewtonBackend.notify_model_changes>` to
  apply queued model-property edits; it ran before ``forward()``, scene-data state reads and steps.
* Added :class:`~isaaclab_newton.physics.newton_backend.StepGraph`: each step ran as straight-line operations, and every
  run of consecutive graphable operations replayed from one CUDA graph, with operations that cannot be captured
  running eagerly in between.
