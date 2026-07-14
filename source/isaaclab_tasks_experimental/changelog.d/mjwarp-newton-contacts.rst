Changed
^^^^^^^

* Changed experimental MJWarp task presets to use Newton-generated contacts. Custom task presets should set
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_contacts` to ``False``; use ``True`` only to
  temporarily restore the deprecated internal contact path.
