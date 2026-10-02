* Added whole-environment-step capture: :meth:`~isaaclab_newton.physics.NewtonManager.step` called while the caller
  records a CUDA graph records the prepared step program into that graph instead of capturing its own, so an MDP and
  the physics step can replay as one graph. ``runtime.record_step`` is the functional form; both require
  :meth:`~isaaclab_newton.physics.NewtonManager.prepare` first and reject programs with operations that cannot be
  recorded.
