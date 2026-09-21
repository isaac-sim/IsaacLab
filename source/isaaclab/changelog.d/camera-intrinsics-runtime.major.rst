Changed
^^^^^^^

* **Breaking:** Made camera intrinsic setters update runtime device buffers without authoring USD.
  USD calibration was imported once at initialization; active calibration became independent of USD
  edit targets and layer composition. To persist calibration, configure
  ``PinholeCameraCfg.from_intrinsic_matrix`` when spawning cameras instead of exporting runtime USD.
* Added ``BaseRenderer.update_camera_intrinsics`` for device calibration updates independently of
  pose updates. Custom renderers must implement this method to support runtime calibration changes.
* Imported initial calibration on the CPU and prepared float32/float64 Warp kernels during
  initialization. Runtime conversion, selection, and duplicate-index handling ran on the camera
  device without matrix or index batch readbacks. Accepted NumPy, Torch, and Warp matrices;
  host inputs were uploaded before runtime updates. OpenCV calibration, synchronous scalar
  validation, and the centered, square-pixel projection were preserved.

Fixed
^^^^^

* Rejected intrinsic matrix shape, batch-cardinality, and index errors before changing any camera.
  Repeated camera indices retained the last matrix in the batch.
