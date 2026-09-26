Changed
^^^^^^^

* Unified Torch/Warp timestamped buffers and SDP transform/geometry caches under the same
  freshness contract, keeping pending-work guards separate from cached-value timestamps.
* Replaced sensor host-side generation counters with a pending-work dirty flag while preserving
  per-environment sampling periods, partial resets, and CUDA graph replay.
