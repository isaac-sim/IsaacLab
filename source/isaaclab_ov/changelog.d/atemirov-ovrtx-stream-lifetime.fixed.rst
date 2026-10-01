* Fixed OVRTX render-output reads and attribute writes on Torch's legacy default CUDA stream
  silently skipping their fence. Warp reports that stream as ``0``, which OVRTX reads as
  "no synchronization"; every OVRTX ``cuda_stream`` handoff now encodes it as ``1``.
* Ordered mapped-buffer release after consuming kernels to prevent reuse before extraction
  completed. Since OVRTX 0.4 the native free is asynchronous on OVRTX's copy stream, so the
  release must carry the consuming stream. The call is non-blocking from OVRTX 0.5.0.377615;
  on 0.4.x it host-syncs the Warp stream once per render var.
