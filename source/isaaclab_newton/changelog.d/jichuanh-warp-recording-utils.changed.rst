* Changed Newton to reset the solver and evaluate forward kinematics once after a state write recorded in a Warp
  stage, inside that stage's CUDA graph, instead of before every physics step once any write was recorded. State
  written inside a raw CUDA graph capture still evaluates them before every physics step
  (:attr:`~isaaclab_newton.physics.NewtonBackend.fk_on_every_step`).
* Changed :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change` inside a recording of
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` to register the change again on each replay, so the next step
  refreshes the solver once for all changes; any other capture still records the refresh of that change.
