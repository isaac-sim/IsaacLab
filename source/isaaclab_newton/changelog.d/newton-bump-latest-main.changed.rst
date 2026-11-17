* Updated the Newton dependency to the current ``main`` revision, which adds the optional OptiX camera backend.
  Newton renamed Kamino's DVI forward-dynamics solver to SCC, so
  :class:`~isaaclab_newton.physics.KaminoDVISolverCfg` now configures Newton's SCC solver with unchanged fields.
  Its :meth:`~isaaclab_newton.physics.KaminoDVISolverCfg.to_solver_config` returns ``dynamics_solver="scc"``;
  Newton keeps ``config.dvi`` as a deprecated alias of ``config.scc``, so read ``config.scc`` instead.
* Changed :attr:`~isaaclab_newton.physics.NewtonCfg.deterministic_mode` ``"gpu_to_gpu"`` to raise during solver
  initialization with MJWarp or Steel, which Newton now limits to ``RUN_TO_RUN``. Use ``"run_to_run"``, or XPBD
  or Featherstone for the cross-GPU guarantee.
