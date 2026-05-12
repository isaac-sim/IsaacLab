Changed
^^^^^^^

* Construct the underlying OVRTX ``Renderer`` in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` ``__init__`` instead of
  during :meth:`~isaaclab_ov.renderers.OVRTXRenderer.prepare_stage`. This
  pairs with the new pre-physics ``__init__`` /
  post-physics :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.initialize`
  lifecycle: when invoked eagerly via
  :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers`, the OVRTX
  ``Renderer`` is created before
  :meth:`~isaaclab.sim.SimulationContext.reset` (and therefore before
  ovphysx initialises), which OVRTX 0.3 requires.
* Replaced an ``assert`` on the OVRTX ``Renderer`` construction with an
  explicit :class:`RuntimeError` so the failure is reported even when
  Python is run with ``-O``.
* Renamed the internal ``OVRTXRenderer.initialize(spec)`` helper to
  ``_initialize_from_spec(spec)`` to avoid shadowing the new
  no-arg :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.initialize`
  lifecycle hook.
