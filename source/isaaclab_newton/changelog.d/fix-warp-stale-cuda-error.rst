Fixed
^^^^^

* Fixed simulation (re)initialization aborting with a spurious ``Warp CUDA
  error 1: invalid argument`` on the first buffer copy when a prior
  simulation lifecycle left a stale CUDA error latched (Warp does not clear
  the error when adding a graph memory free node fails).
  :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation` now drains
  any stale device error before dispatching initialization callbacks.
