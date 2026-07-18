Changed
^^^^^^^

* Changed the ``newton_mjwarp`` launcher backend to use Newton-generated contacts with MJWarp. Custom MJWarp
  configurations should set :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_contacts` to ``False``;
  use ``True`` only to temporarily restore the deprecated internal contact path.
