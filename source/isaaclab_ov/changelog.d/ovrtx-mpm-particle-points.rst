Fixed
^^^^^

* Fixed OVRTX rendering to stream Newton MPM particle positions into registered
  ``UsdGeom.Points`` prims, so kitless cameras can visualize MPM particle clouds.
* Fixed MPM particle clouds silently disappearing in OVRTX because the per-frame
  ``points`` update was written from a GPU buffer, which OVRTX does not use to refresh
  ``UsdGeom.Points`` sphere geometry. Particle positions are now written from host memory
  (deformable meshes keep the zero-copy GPU path).
