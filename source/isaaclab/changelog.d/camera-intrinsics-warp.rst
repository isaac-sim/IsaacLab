Changed
^^^^^^^

* Moved camera intrinsic conversion and matrix-buffer updates into batched Warp kernels. Removed
  the full USD readback after setting pinhole intrinsics and repeated per-camera parameter-name
  conversion. USD attribute authoring remained on the host, with one transfer of converted parameters
  per batch. OpenCV calibration, composed USD opinions, and the square-pixel, centered projection
  behavior were preserved; writes through weaker or mapped edit targets still refreshed from USD.
* Aggregated unsupported pixel-aspect and principal-point warnings once per calibration call.

Fixed
^^^^^

* Rejected intrinsic matrix shape and batch-size mismatches before changing any camera, preventing
  silent partial updates. Repeated camera indices retained the last matrix in the batch.
