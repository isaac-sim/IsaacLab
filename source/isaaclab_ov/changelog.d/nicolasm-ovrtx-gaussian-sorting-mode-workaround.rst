Fixed
^^^^^

* Fixed Gaussian splats rendering as empty tiles in the multi-camera OVRTX PPISP demo by forcing
  ``sortingModeHint = "cameraDistance"`` on its Gaussian splat prims. This temporary workaround for
  ``OMPE-108417`` deliberately overrides the value authored in the asset, so Gaussian splats may
  blend in a different order than the capture requested. It will be removed once the renderer issue
  is fixed.
