* Changed Newton to reset the solver and evaluate forward kinematics once after a state write recorded in a Warp
  stage, inside that stage's CUDA graph, instead of before every physics step once any write was recorded. State
  written inside a raw CUDA graph capture still evaluates them before every physics step
  (:attr:`~isaaclab_newton.physics.NewtonBackend.fk_on_every_step`).
