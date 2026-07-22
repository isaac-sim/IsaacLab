Added
^^^^^

* Added :class:`~isaaclab_newton.visual.NewtonShapeColorWriter`, the kit-less Newton-Warp backend for
  :class:`~isaaclab.envs.mdp.randomize_visual_color`. It applies per-environment diffuse colors by
  writing rows of ``model.shape_color`` (the live render array, no notify) and skips collision shapes.
