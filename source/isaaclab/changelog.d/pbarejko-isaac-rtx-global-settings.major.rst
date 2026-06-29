Changed
^^^^^^^

* **Breaking:** Removed ``RenderCfg`` and ``SimulationCfg.render``. Configure
  Isaac RTX quality settings through
  :class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg` on
  :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.global_settings` instead.
