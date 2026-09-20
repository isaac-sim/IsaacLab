Fixed
^^^^^

* Rejected camera intrinsic matrix batches whose size did not match the selected camera IDs,
  preventing silent partial calibration updates. Callers must provide one matrix per selected
  camera or narrow ``env_ids`` to match the supplied matrices.
