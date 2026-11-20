* **Breaking:** Removed ``physics.ovphysx_compat``, ``ovstage_compat``, and
  ``renderers.ovrtx_compat`` and their older SDK fallbacks. Callers must use the
  pinned SDK APIs, typed handles, and named segmentation tensors directly.
  PhysX schema discovery moved to ``physics.schemas.physx_schema_paths()``.
* Removed OVRTX workarounds for eco-mode GPU transform updates and reusing runtime
  mesh paths, and adopted the current ``MinimalRendering`` mode token.
* Enabled native cloning for acceleration-spring contact materials after the
  OvPhysX 0.8 material-cloning fix.
