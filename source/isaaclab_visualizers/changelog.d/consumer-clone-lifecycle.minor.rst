Changed
^^^^^^^

* Declared Newton-backed visualizer representations before cloning and initialized viewers afterward. Kit streaming
  views acquired the renderer for a configured generated camera from the simulation backend registry before cloning.
  Custom visualizers that pre-register camera renderers should use ``sim.get_or_create_backend(renderer_cfg)``.
