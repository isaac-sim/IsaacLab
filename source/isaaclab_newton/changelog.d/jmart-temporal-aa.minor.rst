Added
^^^^^

* Added :meth:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.provides_temporal_camera_data`
  override that returns ``False``, since the Warp rasterizer does not perform
  temporal accumulation.
