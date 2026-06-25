Added
^^^^^

* Added ``"normals"`` to the rendering correctness validation suite. The ``normals`` data type
  (float32 surface normal vectors in ``[-1, 1]``) is now included in
  :data:`~rendering_test_utils._DEFAULT_SENSOR_DATA_TYPES` for all RTX-based renderer
  combinations and as an explicit parameter for the Newton Warp renderer.
