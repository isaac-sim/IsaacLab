Fixed
^^^^^

* Replaced the fixed 180-degree OpenCV fisheye ray limit with the camera's configured
  ``max_fov``. Calibrated rays beyond the forward hemisphere can now be rendered without
  changing the existing default or introducing per-frame ray generation.
