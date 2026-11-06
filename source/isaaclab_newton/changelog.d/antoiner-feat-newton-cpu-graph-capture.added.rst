* Added Warp CPU graph capture of the Newton physics step when
  :attr:`~isaaclab_newton.physics.NewtonCfg.use_cuda_graph` is set, which is the default. It removes Python
  kernel-launch overhead on ``device="cpu"`` for every solver whose
  :meth:`~isaaclab_newton.physics.NewtonSolver.supports_capture` holds there. MPM and Kamino P-ADMM with an iterative
  linear solver keep stepping eagerly on the CPU. Step callbacks registered on the manager must then issue only Warp
  work, because Warp's CPU capture does not replay torch operations or host reads; set ``use_cuda_graph=False`` to keep
  stepping eagerly on the CPU.
