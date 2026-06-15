Added
^^^^^

* Added :func:`~isaaclab.sim.utils.newton_model_utils.replace_newton_builder_shape_colors` to
  propagate USD material and ``displayColor`` values into a Newton ``ModelBuilder``'s shape colors
  before clone replication, so cloned environments inherit correct colors without a separate
  post-finalize pass.
