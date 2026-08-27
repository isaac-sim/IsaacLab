Fixed
^^^^^

* Fixed the OVRTX renderer re-deriving its CUDA device per call site from the device string, which
  split a bare ``"cuda"`` across GPUs on multi-GPU processes: render-product device ids parsed it
  to device 0 while Warp resolved kernel launches and sync streams on its current CUDA device. The
  renderer now resolves the Warp device once when the render spec arrives, normalizes its device
  string from it, and derives the render-product device ids and every CUDA sync stream — attribute
  writes on both the legacy and ovstage paths, and render-var reads — from the cached device.

* Updated ``map_attribute_for_warp_writes`` to accept a resolved Warp device as well as its string
  alias, preserving the release branch's mapped-write path while keeping its mapping and stream on
  the same device.
