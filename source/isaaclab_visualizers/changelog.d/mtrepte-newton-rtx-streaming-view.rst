Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizer` unconditionally reporting the
  streaming/tiled camera view as unsupported. Setting ``streaming_view=True`` now creates the owned
  streaming camera sensor and produces composites via ``render_tiled_rgb_array()``, usable for headless
  capture (e.g. through :class:`~isaaclab.envs.VideoRecorderCfg`). The live on-screen streaming preview
  panel remains unavailable on this backend, since ``ViewerRTX.log_image`` has no display sink.
