Fixed
^^^^^

* Fixed locomanipulation SDG generation with NuRec backgrounds by enabling camera capture, applying Isaac RTX
  Gaussian renderer settings, syncing randomized fixture poses, and recording the projected scene state after
  placement. ``--high_res_video`` now records RGB observations at 512x320 instead of 960x540; update MP4
  conversion dimensions and model input shapes accordingly.
