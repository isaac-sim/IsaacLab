Fixed
^^^^^

* Fixed ``SceneDataProvider`` allocating transform and geometry buffers on Warp's process-global
  default device. Allocations now follow the device of the publication they consume.
* Selected the configured PyTorch and Warp process device in ``SimulationContext`` before
  constructing physics, rendering, and visualization backends.
