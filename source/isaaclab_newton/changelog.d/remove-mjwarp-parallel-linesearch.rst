Deprecated
^^^^^^^^^^

* Deprecated the MuJoCo Warp parallel line search config option. Setting it
  emits a warning and is ignored; use
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.ls_iterations` to tune the
  iterative line search path.
