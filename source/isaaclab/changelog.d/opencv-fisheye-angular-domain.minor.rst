Added
^^^^^

* Added a configurable ``OpenCvFisheyeDistortionCfg.max_fov`` angular domain for Newton,
  preserving the existing 180-degree default. Documented that RTX/OVRTX's native OpenCV
  schema does not consume this setting and can clip rectangular sensor corners to its image circle.
