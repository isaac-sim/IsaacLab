Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.utils.newton_model_utils.replace_newton_shape_colors` incorrectly
  keeping Newton's random palette colors for environments 1–N when USD cloning is skipped
  Shape labels for non-source environments now resolve to the clone-plan source prim so
  the correct USD material color is applied to all environments.
