* Added Warp CPU graph capture of the Newton physics step for the MJWarp (without
  ``use_mujoco_cpu``), XPBD and Featherstone solvers when :attr:`~isaaclab_newton.physics.NewtonCfg.use_cuda_graph`
  is set, which is the default. It removes Python kernel-launch overhead on ``device="cpu"``. Solvers opt in through
  the new :attr:`~isaaclab_newton.physics.NewtonSolver.cpu_capture` flag; the others keep stepping eagerly on the
  CPU. Step callbacks registered on the manager must then issue only Warp work, because Warp's CPU capture does not
  replay torch operations or host reads; set ``use_cuda_graph=False`` to keep stepping eagerly on the CPU.
