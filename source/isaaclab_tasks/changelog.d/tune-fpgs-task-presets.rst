Changed
^^^^^^^

* Changed first-party MJWarp task presets to use Newton-generated contacts and removed obsolete internal-contact CCD
  tuning. Custom task presets should set :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_contacts` to
  ``False``; use ``True`` only to temporarily restore the deprecated internal contact path.

Fixed
^^^^^

* Fixed task-owned FeatherPGS presets, including A1, Go1, and H1 locomotion, to use validated constraint capacities and
  validated single-substep H1 flat execution. Reward and throughput acceptance remain task-specific.
