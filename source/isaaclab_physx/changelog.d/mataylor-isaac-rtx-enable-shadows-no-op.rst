Fixed
^^^^^

* Fixed :attr:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg.enable_shadows`
  documenting itself as a working shadow switch. It writes ``/rtx/shadows/enabled``, which the RTX
  version shipped by Isaac Sim 6.0 registers but no render mode reads, so setting it to ``False``
  leaves every camera output unchanged. The docstring now records the limitation and points at
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.enable_shadows`, which does take effect.
