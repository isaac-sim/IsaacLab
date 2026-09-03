Added
^^^^^

* Added translation of :attr:`~isaaclab.physics.PhysicsCfg.deterministic` in ``NewtonManager``.
  The request selects ``deterministic_mode="run_to_run"`` and sets ``MJWarpSolverCfg.disable_sensors``
  on the MJWarp GPU path. An explicitly set ``deterministic_mode`` takes precedence. MuJoCo on the
  CPU is left unchanged: Warp's deterministic mode does not reach that path, and the request is
  logged instead of applied.

Fixed
^^^^^

* Fixed a determinism request silently starving the IMU, PVA, and joint-wrench sensors. Disabling
  MuJoCo Warp's sensors also skips the ``rne_postconstraint`` stage that fills ``body_qdd`` and
  ``body_parent_f``, so those sensors reported stale values. ``NewtonManager`` now raises at solver
  initialization when a scene requests both.
