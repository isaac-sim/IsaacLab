* Fixed lazily computed asset data properties, such as
  :attr:`~isaaclab_newton.assets.ArticulationData.projected_gravity_b`, inside a CUDA graph recorded by
  :class:`~isaaclab.utils.warp.capture.GraphedCallable`: 15 of them raised while a graph recorded, and the others
  replayed the value of the recording step when they were current at recording. A recording now holds their
  update, so every replay recomputes them.
* Fixed forward kinematics after a state write recorded by :class:`~isaaclab.utils.warp.capture.GraphedCallable`
  being skipped on replays when an :func:`~isaaclab.utils.warp.capture.eager` call of the same recording read body
  state after the write: every replay now evaluates them right before that call.
