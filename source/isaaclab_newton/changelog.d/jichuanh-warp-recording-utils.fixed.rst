* Fixed model changes registered with :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change` inside a CUDA
  graph recording, such as fixed tendon property writes, reaching the solver only when the graph was recorded: a part
  of the step recorded with :class:`~isaaclab.utils.warp.capture.GraphedCallable` registers the change again on each
  replay, and any other capture records the solver's refresh.
