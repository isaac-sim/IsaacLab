Added
^^^^^

* Added declarative ``cloning_contexts`` to renderer and visualizer configurations and routed their representations
  through the shared clone plan. Visualizers were constructed before cloning and initialized after physics was ready.

Removed
^^^^^^^

* **Breaking:** Removed ``RenderContext.get_renderer`` and consolidated renderer ownership in the simulation backend
  registry. Replace ``sim.render_context.get_renderer(renderer_cfg)`` with
  ``sim.get_or_create_backend(renderer_cfg)``. ``RendererCfg`` extended ``BackendCfg``; ``RenderContext`` retained
  rendering lifecycle coordination without a separate renderer cache or ownership. Use ``sim.render_context``
  instead of constructing a standalone ``RenderContext()``, whose constructor now requires the simulation registry.
