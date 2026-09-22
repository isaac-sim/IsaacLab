Added
^^^^^

* Added ``hardening_rate`` and ``softening_rate`` to
  :class:`~isaaclab_newton.sim.MPMParticleMaterialCfg`.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.sim.MPMGridCfg` and
  :class:`~isaaclab_newton.sim.MPMPointsCfg` to author schema-valid USD points,
  explicit particle masses, and bound Newton MPM materials for Newton's standard
  USD import path.
* Standardized :class:`~isaaclab_newton.physics.MPMSolverCfg` rheology solver
  values on the canonical tokens defined by ``NewtonMPMSceneAPI``.
* **Breaking:** Standardized grid jitter as one deterministic asset-local particle
  distribution shared by USD clones. Use reset events or domain randomization for
  independent per-environment distributions.
* Updated the teapot-fill water material's tensile yield ratio to the schema-valid
  maximum of ``1.0``.

Removed
^^^^^^^

* **Breaking:** Removed ``emit_mpm_particles``. Use ``MPMGridCfg`` or
  ``MPMPointsCfg`` through the standard Isaac Lab spawner workflow.
