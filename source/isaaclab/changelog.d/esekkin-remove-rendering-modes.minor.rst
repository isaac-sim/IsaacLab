Removed
^^^^^^^

* **Breaking:** Removed the ``performance``, ``balanced``, and ``quality`` RTX rendering mode presets, along with
  the ``--rendering_mode`` CLI argument and the :attr:`~isaaclab.sim.RenderCfg.rendering_mode` field.
  High-fidelity RTX defaults (matching the former ``quality`` preset) now live in the rendering experience
  files (``apps/isaaclab.python.rendering.kit`` and ``apps/isaaclab.python.headless.rendering.kit``) and are
  applied automatically when camera rendering is enabled; they can still be customized via
  :class:`~isaaclab.sim.RenderCfg`. For high-performance rendering, use the RTX Minimal renderer instead.
  To migrate, drop ``--rendering_mode`` / ``RenderCfg(rendering_mode=...)`` and override individual settings
  via :class:`~isaaclab.sim.RenderCfg` if needed.

Changed
^^^^^^^

* Moved the RTX-specific knowledge out of :class:`~isaaclab.sim.SimulationContext`. The RTX defaults
  previously held in a core dictionary now live in the rendering experience files, and the
  :class:`~isaaclab.sim.RenderCfg` field-to-carb-path mapping moved to the ``isaaclab_physx`` backend.
  :meth:`~isaaclab.sim.SimulationContext._apply_render_cfg_settings` now only owns the apply timing and
  delegates to ``isaaclab_physx.renderers.apply_rtx_render_settings`` when available.
