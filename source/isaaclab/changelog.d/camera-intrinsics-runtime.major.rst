Changed
^^^^^^^

* **Breaking:** Made camera intrinsic setters update runtime device buffers without authoring USD.
  USD calibration was imported once at initialization; active calibration became independent of USD
  edit targets and layer composition. To persist calibration, configure
  ``PinholeCameraCfg.from_intrinsic_matrix`` when spawning cameras instead of exporting runtime USD.
* Added ``BaseRenderer.update_camera_intrinsics`` for device calibration updates independently of
  pose updates. Custom renderers must implement this method to support runtime calibration changes.
* Batched intrinsic conversion, selection, and duplicate-index handling in Warp. Matrix and index
  batches stayed on device; one scalar status readback retained synchronous validation and warnings.
  OpenCV calibration and the centered, square-pixel pinhole projection were preserved.

Fixed
^^^^^

* Rejected intrinsic matrix shape, batch-cardinality, and index errors before changing any camera.
  Repeated camera indices retained the last matrix in the batch.
