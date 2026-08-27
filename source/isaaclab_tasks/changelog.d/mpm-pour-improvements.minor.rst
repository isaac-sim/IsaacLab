Changed
^^^^^^^

* Made ``IsaacContrib-Franka-Pour`` configure its initial particle lattice by
  source-cup fill height independently of the reset-state artifact.
* Replaced the receiver's rigid-only mesh collider with an analytic box while
  retaining its hollow mesh for MPM particle collisions.
* Expressed the reference 735-particle lattice with the same 15 mm voxel as the
  MPM solver and three particles per cell along each axis.

Fixed
^^^^^

* Matched particle-count and sparse-grid capacity calculations to the MPM
  spawner's per-axis ceiling behavior.
* Stabilized the taller default particle payload with two MPM entry substeps
  and particle-backed automatic warm starting, and increased the proxy mass
  scale to prevent unphysical rigid-cup recoil.
