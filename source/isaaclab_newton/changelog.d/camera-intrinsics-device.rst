Fixed
^^^^^

* Regenerated Newton camera rays in place after runtime calibration changes, preserving buffers used
  by captured rendering graphs. Pinhole ray generation used Warp directly without scalar GPU-to-host
  transfers through the native convenience helper.
* Rejected runtime calibration batches that differed across environments before changing active
  calibration. Newton's native ray field was shared across worlds and previously used only the
  first camera's calibration. Use identical intrinsics across environments with this renderer.
