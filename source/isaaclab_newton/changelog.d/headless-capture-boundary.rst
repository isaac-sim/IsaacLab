Fixed
^^^^^

* Preserved standard CUDA graph capture in headless Kit physics sessions without rendering, avoiding
  an extra simulation warmup step while retaining relaxed capture for GUI and offscreen rendering.
