Removed
^^^^^^^

* **Breaking:** Removed the ``Renderer`` and ``Visualizer`` factory classes and
  ``VisualizerCfg.create_visualizer``. Concrete renderer and visualizer configs now declare
  ``class_type``; construct custom implementations as ``cfg.class_type(cfg)`` and acquire camera
  renderers through ``SimulationContext.render_context.get_renderer(cfg)``. Replaced the removed
  ``VisualizerCfg.get_visualizer_type()`` accessor with the ``visualizer_type`` field.
