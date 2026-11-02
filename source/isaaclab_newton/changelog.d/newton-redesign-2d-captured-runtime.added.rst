* Added ``NewtonBackend.live``, true while a step runs or a CUDA graph is recorded. Derived asset data recomputed on
  read in that interval instead of using its timestamp cache.
* Added :meth:`NewtonBackend.notify_model_changes <isaaclab_newton.physics.NewtonBackend.notify_model_changes>` to
  apply queued model-property edits; it ran before ``forward()``, scene-data state reads and steps.
