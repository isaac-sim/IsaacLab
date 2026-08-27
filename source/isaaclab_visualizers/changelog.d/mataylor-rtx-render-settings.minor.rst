Added
^^^^^

* Added :attr:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg.render_settings`, which authors arbitrary RTX
  attributes onto the OVRTX render product as ``{name: (usd_type_name, value)}``. ``ViewerRTX`` hard-codes its render
  product and exports the stage before the renderer reads it, so these are applied in the only window that reaches the
  renderer. For example, ``{"omni:rtx:quality:minSpp": ("Int", 32)}`` raises the path tracer's sample floor, which
  ``ViewerRTX`` otherwise leaves low to keep interactive latency down.
