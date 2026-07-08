Changed
^^^^^^^

* **Breaking:** Removed
  :attr:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg.rendering_mode`
  and the ``performance``, ``balanced``, and ``quality`` RTX preset files.
  Override individual settings through
  :class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg` fields or
  ``carb_settings`` instead.
