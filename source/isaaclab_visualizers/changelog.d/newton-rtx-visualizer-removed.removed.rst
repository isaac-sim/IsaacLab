* **Breaking:** Removed ``NewtonRTXVisualizer``, ``NewtonRTXBackendCfg`` and ``NewtonRTXVisualizerCfg``. The Newton
  visualizer now renders with OpenGL or RTX: use :class:`~isaaclab_visualizers.newton.NewtonGLBackendCfg` and set
  :attr:`~isaaclab_visualizers.newton.NewtonGLBackendCfg.initial_renderer` to ``"rtx"`` to start in RTX. Move
  ``renderer_cfg`` and ``render_settings`` to ``rtx_renderer_cfg`` and ``rtx_render_settings``.
