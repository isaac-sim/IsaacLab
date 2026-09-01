Added
^^^^^

* Added :attr:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg.render_settings`, which authors arbitrary RTX
  attributes onto the OVRTX render product as ``{name: (usd_type_name, value)}``. ``ViewerRTX`` hard-codes its render
  product and exports the stage before the renderer reads it, so these are applied in the only window that reaches the
  renderer. For example, ``{"omni:rtx:quality": ("Int", 100)}`` re-enables the path tracer's quality convergence
  loop, which ``ViewerRTX`` otherwise disables to keep interactive latency down.
