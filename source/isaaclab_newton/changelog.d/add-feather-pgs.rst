Added
^^^^^

* Added :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg` and
  :class:`~isaaclab_newton.physics.NewtonFeatherPGSManager` for Newton's
  reduced-coordinate FeatherPGS solver.

Changed
^^^^^^^

* Changed the default FeatherPGS position solve from twelve to eight iterations
  so task families share the measured large-batch baseline. Set
  ``pgs_iterations=12`` explicitly to retain the previous behavior.
* Changed the default FeatherPGS position-correction factor from ``0.2`` to
  ``0.05`` for stable shared task presets. Set ``pgs_beta=0.2`` explicitly to
  retain the previous correction strength.
* Changed the default FeatherPGS velocity-limit activation fraction from
  ``0.0`` to ``0.7`` so inactive limit rows do not consume solver work. Set
  ``velocity_limit_activation_fraction=0.0`` to retain always-allocated rows.
* Changed FeatherPGS CUDA-graph capture to seed asynchronous double-buffer
  events before captured solver work.
* Added ``pgs_inner_substeps`` to expose Newton's opt-in frozen-basis
  position-solve and integration cycles.

Fixed
^^^^^

* Fixed the default FeatherPGS joint-limit activation gap to avoid allocating
  inactive lower and upper limit rows on every simulation step.
