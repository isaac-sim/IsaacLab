Added
^^^^^

* Added the manager-based ``IsaacContrib-Rizon-Sharpa-Coffee`` environment with
  procedural tableware, coupled Newton MPM/MJWarp physics, a two-scoop IK demo,
  optional stirring, Apple Vision Pro retargeting and named grasp-reset authoring.
  The default uses a measured-palm tool attachment; free-spoon contact grasping
  remains experimental. The licensed robot bundle must be provided separately.
* Added mass-preserving soluble-coffee conversion into liquid during captured
  physics, world-local material resets and conservative colour transport.
  Optional GL, RTX and Kit presentation includes native rendering grains,
  reconstructed liquid, a following camera and simulation-time MP4 capture.

Fixed
^^^^^

* Tapered native spoon collider margins near the rim while retaining broad-face
  protection and the original shallow spoon geometry.
* Stabilized liquid presentation with a render-only temporal density filter,
  shared-vertex colour interpolation and reconstruction caching per physics state.
  These changes use existing Newton surface APIs without modifying Newton itself.
