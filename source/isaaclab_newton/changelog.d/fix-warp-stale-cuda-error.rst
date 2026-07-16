Fixed
^^^^^

* Fixed simulation (re)initialization aborting with a spurious ``Warp CUDA
  error 1: invalid argument`` on the first buffer copy when a garbage
  collection pass inside a CUDA graph capture freed a graph-scoped
  allocation while the capture was paused for a conditional body (Warp
  latches the failed free's error without clearing it). Garbage collection
  is now paused for the duration of graph capture, and
  :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation` drains
  any stale device error before dispatching initialization callbacks.
