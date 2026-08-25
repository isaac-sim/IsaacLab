Changed
^^^^^^^

* Changed the OVRTX ovstage path to write object transforms, camera transforms and deformable or
  particle points straight from their Warp GPU buffers as CUDA DLTensors, removing the per-frame
  host copies that ``ovstage 0.1.0`` required.
* Changed those writes to be ordered by handing ovstage the producing Warp stream
  (``write_attribute(cuda_stream=...)``), replacing the device-wide ``wp.synchronize_device`` with
  stream-scoped producer ordering, and matching the legacy OVRTX binding path. The write is still
  awaited, so the calling thread can block; the gain is the removed host copy and the narrower
  synchronization scope, not a nonblocking handoff.
