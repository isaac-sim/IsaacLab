Added
^^^^^

* Added :class:`~isaaclab.sim.spawners.sensors.OpenCvPinholeDistortionCfg` and
  :class:`~isaaclab.sim.spawners.sensors.OpenCvFisheyeDistortionCfg` and a ``distortion`` field on
  :class:`~isaaclab.sim.spawners.sensors.PinholeCameraCfg`, letting a camera carry an OpenCV
  ``fx/fy/cx/cy`` + distortion-coefficient calibration. The model is authored on the camera prim as
  the ``omni:lensdistortion:*`` USD API, which the RTX/OVRTX renderer honors natively.

Changed
^^^^^^^

* Changed the reconstructed :attr:`~isaaclab.sensors.camera.Camera.data` intrinsic matrices to use the
  authored OpenCV ``fx/fy/cx/cy`` when a lens-distortion model is present, so they reflect a real
  calibration (non-square pixels or an off-center principal point) instead of assuming ``fx == fy`` and
  a centered principal point.
