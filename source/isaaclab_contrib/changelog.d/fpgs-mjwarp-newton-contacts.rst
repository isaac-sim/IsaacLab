Changed
^^^^^^^

* Changed first-party coupled MJWarp-VBD configurations to share Newton-generated contacts between their rigid and
  soft sub-solvers. New configurations should set
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_contacts` to ``False``; the public default remains
  ``True`` during the deprecation window for compatibility with the MuJoCo internal contact path.
