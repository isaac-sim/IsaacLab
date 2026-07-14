Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` ignoring the camera's far clipping
  plane. The renderer now uses ``spawn.clipping_range[1]`` as the Newton sensor ``max_distance`` per
  camera instead of the fixed :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.max_distance`
  default, so rays are clipped at the configured far plane.

Added
^^^^^

* Added :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.depth_clipping_behavior` to
  :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`, mirroring the RTX renderer. ``"max"``
  replaces the ``0.0`` background (missed rays / beyond the far plane) with the far clip distance;
  ``"none"`` and ``"zero"`` leave it at ``0.0``.
