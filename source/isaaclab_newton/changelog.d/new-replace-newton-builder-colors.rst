Changed
^^^^^^^

* Moved Newton shape color propagation from post-finalize (on the model) to pre-clone (on the
  builder) in :class:`~isaaclab_newton.physics.NewtonManager` and the cloner utilities. Colors are
  now set via :func:`~isaaclab.sim.utils.newton_model_utils.replace_newton_builder_shape_colors`
  before ``ModelBuilder`` replication, so all cloned environments automatically inherit the correct
  USD material colors without an extra GPU scatter pass after finalization.
