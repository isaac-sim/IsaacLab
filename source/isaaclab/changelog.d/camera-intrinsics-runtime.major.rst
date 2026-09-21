Changed
^^^^^^^

* **Breaking:** Made camera intrinsic setters update runtime device buffers without authoring USD.
  USD calibration was imported once at initialization; active calibration became independent of USD
  edit targets and layer composition. To persist calibration, configure
  ``PinholeCameraCfg.from_intrinsic_matrix`` when spawning cameras instead of exporting runtime USD.
* Added ``BaseRenderer.update_camera_intrinsics`` for device calibration updates independently of
  pose updates. Custom renderers must implement this method to support runtime calibration changes.
* Prepared intrinsic conversion, selection, and duplicate-index handling in NumPy, avoiding camera
  calibration kernel compilation for setup-time updates. Accepted NumPy, Torch, and Warp matrices;
  device inputs were copied to the host before native renderer updates consumed the device buffers.
  OpenCV calibration, synchronous validation, and the centered, square-pixel projection were preserved.

Fixed
^^^^^

* Rejected intrinsic matrix shape, batch-cardinality, and index errors before changing any camera.
  Repeated camera indices retained the last matrix in the batch.
