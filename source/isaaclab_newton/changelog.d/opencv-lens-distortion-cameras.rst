Added
^^^^^

* Added a warning when a camera cfg carries an OpenCV lens-distortion model
  (``spawn.distortion``) under the Newton renderer, which does not yet apply the model; the camera
  renders undistorted. The distortion cfg is renderer-agnostic and remains a documented extension
  point for a future Newton implementation.
