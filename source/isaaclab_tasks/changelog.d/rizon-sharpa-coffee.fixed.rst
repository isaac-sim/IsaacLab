* Tapered native spoon collider margins near the rim while retaining broad-face
  protection and the original shallow spoon geometry.
* Stabilized liquid presentation with a render-only temporal density filter,
  shared-vertex colour interpolation and reconstruction caching per physics state.
  These changes use existing Newton surface APIs without modifying Newton itself.
  The liquid mesh resolution was capped at 1.5 mm when coarsening the physics grid.
* Raised and widened the initial water fill and replaced the cup's tangential
  handle with a curved handle attached to the wall at both ends. Shared vessel
  glaze authoring and surface-cache invalidation removed repeated setup code.
* Bounded particle advection by voxel size and the MPM substep duration to retain
  collider queries when using coarser timesteps with a fuller water fill.
* Reduced liquid contraction during stirring with Newton's existing standard
  Gauss-Seidel solver, a larger wet-run pressure iteration budget and a 400 Hz
  default for mounted wet scripts at 3 mm or coarser resolution.
