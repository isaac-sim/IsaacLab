* Routed independently bound PreviewSurface color updates through constant geometry primvars in the
  OVRTX legacy API's exported scene. Shared bindings and other rendering consumers retained their
  existing behavior. Existing :class:`~isaaclab.assets.VisualMaterial` configurations required no changes.
  Fully GPU-resident rendering updates still required native OVRTX support for GPU constant-color primvars.
