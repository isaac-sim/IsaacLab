Fixed
^^^^^

* Fixed the OVRTX renderer re-deriving its CUDA device per call site from the device string, which
  split a bare ``"cuda"`` across GPUs on multi-GPU processes: render-product device ids parsed it
  to device 0 while Warp resolved kernel launches and sync streams on its current CUDA device. The
  renderer now resolves the Warp device once when the render spec arrives, normalizes its device
  string from it, and derives the render-product device ids and every CUDA sync stream — attribute
  writes on both the legacy and ovstage paths, and render-var reads — from the cached device.

Removed
^^^^^^^

* Removed the ``isaaclab_ov.renderers.ovrtx_mapping`` module
  (:func:`map_attribute_for_warp_writes` and ``cuda_device_id``). Nothing calls it since GPU
  transform updates moved to caller-owned buffers with
  ``binding.write(data_access=DataAccess.ASYNC, cuda_stream=...)``, and mapping OVRTX attribute
  memory for per-frame GPU writes is an anti-pattern: every map/unmap cycle is a hidden
  ``cudaMalloc``/``cudaFree``, and the API is deprecated in ovrtx and refused in BORROW attach
  mode. Migration: write into a persistent caller-owned Warp buffer and hand it to
  ``binding.write(..., cuda_stream=<producing Warp stream>)``; if mapping is unavoidable, pass the
  producing stream explicitly via ``unmap(stream=...)`` — the mapping's context manager commits
  without any CUDA sync.
