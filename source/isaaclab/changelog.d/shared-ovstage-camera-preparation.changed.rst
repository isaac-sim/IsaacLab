* Resolved ``CameraCfg.isp_cfg`` and authored RTX exposure overrides during camera construction, before
  scene cloning or native import, including cameras with ``spawn=None``. Authored PPISP settings must be
  present before constructing the sensor. Renderers now consumed the prepared configuration and retained
  their normal runtime initialization; registry-owned renderers remained available across sensor resets.
* **Breaking:** Removed the redundant ``BaseRenderer.prepare_cameras`` phase. Custom renderers should perform per-camera
  runtime setup in ``create_render_data`` and access the authored stage through ``self._sim.stage``.
