Changed
^^^^^^^

* Changed the buffer update of the contact sensor to capture its warp kernels into CUDA
  graphs and replay them on subsequent updates, reducing the per-step CPU overhead of the
  sensor. The PhysX tensor reads still run eagerly since they cannot be graph-captured.
  The kernels also run eagerly on CPU devices or when graph capture fails. Updating the
  sensor while an outer CUDA graph capture is active now raises an error, since replays
  of such a graph would consume stale contact data.
