Fixed
^^^^^

* Fixed ``--deterministic`` not making camera observations reproducible. The flag configured the
  physics solver but left :attr:`warp.config.deterministic` at ``NOT_GUARANTEED``, and Newton's
  sensor and geometry kernels -- unlike its solvers -- take no per-module determinism option and
  fall back to that global. The scene BVH is built over an atomically compacted shape list, so its
  primitive order varied between processes and a tiled camera rendered a few pixels differently
  from identical simulation state, which was enough to make image-observation training diverge.
