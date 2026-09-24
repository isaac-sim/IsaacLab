Fixed
^^^^^

* Resolved deformable geometry from declared prototypes during backend import, preserving
  custom namespaces and partial environment coverage without storing geometry on ``ClonePlan``.
* Added versioned SDP geometry publications for deformables, particles, and cables, with native
  point views or one fused interpolation and layout conversion. Replaced internal flat-node queries
  with ``get_geometry_points`` using visual prim paths.
* Removed the renderer's geometry step gate so native position writes between renders remain visible.
* Released cached destination bindings with their consumer-owned buffers instead of retaining them
  for the simulation's lifetime.
