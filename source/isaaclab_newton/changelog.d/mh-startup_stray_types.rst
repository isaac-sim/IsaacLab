Changed
^^^^^^^

* Changed Newton cloning to compose per-world transforms in bulk with NumPy and to pre-normalize
  destination prim paths, reducing per-world Python work during scene replication.
* Changed the garbage-collector pause used around CUDA graph capture in
  :class:`~isaaclab_newton.physics.NewtonManager` to collect only generation 0 afterwards,
  avoiding a full-heap walk once the replicated model exists.
