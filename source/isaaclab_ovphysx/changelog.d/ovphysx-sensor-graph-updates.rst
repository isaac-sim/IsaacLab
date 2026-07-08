Changed
^^^^^^^

* Changed the buffer updates of the OvPhysX contact sensor, frame transformer,
  joint wrench sensor, and ray caster to capture their warp kernels into CUDA
  graphs and replay them on subsequent updates, reducing per-step CPU overhead.
  The blocking OvPhysX tensor reads still run eagerly since they cannot be
  graph-captured. Updating any OvPhysX sensor (including the IMU and PVA
  sensors) while an outer CUDA graph capture is active now raises an error,
  since replays of such a graph would consume stale data.
