Added
^^^^^

* Added :class:`~isaaclab_newton.cloner.BatchedModelBuilder` and
  :func:`~isaaclab_newton.cloner.replicate_builder_mapping_batched`, a vectorized
  Newton replication path that appends all cloned environments to the model in one
  batched pass and writes final per-environment labels directly, significantly
  reducing startup time for large environment counts. The legacy per-environment
  path remains the default; opt in via :attr:`NewtonCfg.use_batched_model_builder`
  or the new :func:`~isaaclab_newton.cloner.newton_physics_replicate_batched` entry
  point. Scenes with per-world builder hooks (e.g. MPM/deformable objects)
  automatically fall back to the legacy path.
* Added :func:`~isaaclab_newton.cloner.compare_builder_states` and
  :func:`~isaaclab_newton.cloner.compare_finalized_models` for validating that two
  Newton builders or finalized models are equivalent, with actionable per-field
  mismatch reports.
