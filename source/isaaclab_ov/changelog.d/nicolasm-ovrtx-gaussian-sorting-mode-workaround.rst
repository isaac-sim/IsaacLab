Fixed
^^^^^

* Fixed Gaussian splats rendering as empty tiles on the OVRTX renderer whenever more than one camera
  was rendered, by forcing ``sortingModeHint = "cameraDistance"`` on every Gaussian splat prim in
  :meth:`~isaaclab_ov.renderers.OVRTXRenderer.prepare_stage`. RTX drops all Gaussian contribution, in
  every tile, for the ``zDepth`` sort mode that NuRec exports author (and that RTX itself falls back to
  when the token is absent) once a render product carries more than one camera. This is a temporary
  workaround for the renderer bug and deliberately overrides the value authored in the asset, so
  Gaussian splats may blend in a different order on OVRTX than the capture asked for. It will be
  removed once the renderer is fixed.
