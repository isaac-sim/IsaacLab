Fixed
^^^^^

* Fixed OVRTX scenes with more than one camera failing at startup with
  ``Layout-compatible non-array tensor shape[0] (N) must equal binding prim count (1)``. Cameras
  registered after the first one bound the camera prims authored on the USD stage, which is one
  prototype per spawn variant rather than one per environment whenever USD replication does not
  run, as in kitless runs on OvPhysx and Newton. Every camera now binds one prim per environment,
  for both its transform and its calibration columns.
