Removed
^^^^^^^

* **Breaking:** Removed the ``performance``, ``balanced``, and ``quality`` RTX rendering mode presets, along with
  the ``--rendering_mode`` CLI argument and the :attr:`~isaaclab.sim.RenderCfg.rendering_mode` field.
  High-fidelity RTX defaults (matching the former ``quality`` preset) are now applied automatically and
  can still be customized via :class:`~isaaclab.sim.RenderCfg`. For high-performance rendering, use the
  RTX Minimal renderer instead. To migrate, drop ``--rendering_mode`` /
  ``RenderCfg(rendering_mode=...)`` and override individual settings via
  :class:`~isaaclab.sim.RenderCfg` if needed.
