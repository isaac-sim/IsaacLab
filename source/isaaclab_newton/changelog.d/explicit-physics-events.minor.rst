Added
^^^^^

* Added directly configurable physics randomization terms in ``envs.mdp``.
  Terms borrowed the active simulation's resources and retained event-local state.
  Newton terms exposed a single friction range, native collider margin/gap
  parameters, and per-world gravity. Deprecated core entry points retained their
  previous material signatures and collider-offset conversion during migration.
