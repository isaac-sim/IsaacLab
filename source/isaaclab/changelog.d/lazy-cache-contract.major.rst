Changed
^^^^^^^

* **Breaking:** Removed ``TimestampedBufferWarp``. Use ``TimestampedBuffer(wp.zeros(...))``
  or ``TimestampedBuffer(torch.zeros(...))`` with caller-owned storage instead.
* **Breaking:** Renamed ``SceneDataBackend.transforms_version`` to ``transforms_timestamp``.
  Update custom scene-data producers and consumers to use the new name.
* Unified asset and SDP cache timestamps, keeping pending-work dirty flags separate from
  cached-value timestamps.
* Replaced sensor host-side generation counters with a pending-work dirty flag while preserving
  per-environment sampling periods, partial resets, and CUDA graph replay.
