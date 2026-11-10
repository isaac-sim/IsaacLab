* Fixed lazily computed asset data properties, such as
  :attr:`~isaaclab_newton.assets.ArticulationData.projected_gravity_b`, inside a CUDA graph recorded by
  :class:`~isaaclab.utils.warp.capture.GraphedCallable`: 15 of them raised while a graph recorded, and the others
  replayed the value of the recording step when they were current at recording. A recording now holds their
  update, so every replay recomputes them.
