* **Breaking:** Moved PPISP processing from the renderer to ``CameraCfg.pipeline``.
  Replace ``CameraCfg.isp_cfg`` with ``PpispStepCfg``; use ``rgb_radiance`` for its scene-linear
  input, equivalent to ``rgb_hdr`` on Newton.
