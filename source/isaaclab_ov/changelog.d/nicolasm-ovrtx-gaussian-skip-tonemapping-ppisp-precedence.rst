Fixed
^^^^^

* Fixed the OVRTX renderer forcing Gaussian tonemapping on whenever ``rgb_hdr`` was requested, even
  when it was requested for a reason other than PPISP. An explicit ``OvrtxRendererCfg.gaussian_skip_tonemapping``
  request is now honored unless the camera has PPISP configured (``CameraCfg.isp_cfg``), which still
  takes precedence since it requires tonemapped-off HDR input. Without such a request an ``rgb_hdr``
  camera keeps authoring the attribute off, since OVRTX enables skip-tonemapping by default and that
  renders Gaussian pixels fully black. The precedence decision now lives in ``OVRTXRenderer`` rather
  than the ``ovrtx_usd`` render-scope builder, which now simply authors the value it is given.
* Fixed the Gaussian ``skipTonemapping`` RenderProduct attribute being authored without the
  ``OmniRtxSettingsParticleFieldAPI_1`` schema that gates its namespace, which its
  ``accumulatedAlbedo`` sibling already applied.
* Fixed ``accumulation_limit`` alone authoring ``omni:rtx:rt:accumulation:enabled = false`` on the
  render product, which disabled the accumulation the limit was meant to bound.
