Added
^^^^^

* Added :func:`~isaaclab_physx.renderers.apply_rtx_render_settings`, which translates
  :class:`~isaaclab.sim.RenderCfg` overrides into RTX carb settings. This keeps the RTX-specific
  field-to-carb-path mapping in the ``isaaclab_physx`` backend instead of core
  :class:`~isaaclab.sim.SimulationContext`.
