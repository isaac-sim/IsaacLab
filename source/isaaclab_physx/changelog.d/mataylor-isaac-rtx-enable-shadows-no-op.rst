Fixed
^^^^^

* Fixed :attr:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg.enable_shadows`
  documenting itself as a working shadow switch. It writes ``/rtx/shadows/enabled``, the shadow
  switch of the ``RaytracedLighting`` pipeline, which the RTX version shipped by Isaac Sim 6.0 no
  longer offers; the render modes that remain do not read it, so setting the field to ``False``
  leaves every camera output byte-identical. The docstring now records the limitation and points at
  the ``albedo`` data type, ambient-only lighting, and
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.enable_shadows` as the shadow-free options that do
  work.
