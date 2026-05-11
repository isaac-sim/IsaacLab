Changed
^^^^^^^

* Construct the underlying OVRTX ``Renderer`` in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` ``__init__`` instead of
  during :meth:`~isaaclab_ov.renderers.OVRTXRenderer.prepare_stage` so the
  backend is fully created when registered on the simulation-scoped
  :class:`~isaaclab.renderers.render_context.RenderContext` (e.g. via
  :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers`), front-loading
  setup and logging before the first
  :meth:`~isaaclab.sim.SimulationContext.reset`.
