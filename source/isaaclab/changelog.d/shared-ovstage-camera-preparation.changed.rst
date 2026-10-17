* Resolved ``CameraCfg.isp_cfg`` and authored RTX exposure overrides during camera construction, before
  scene cloning or native import, including cameras with ``spawn=None``. Authored PPISP settings must be
  present before constructing the sensor. Renderers now consumed the prepared configuration and retained
  their normal runtime initialization; registry-owned renderers remained available across sensor resets.
