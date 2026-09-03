Added
^^^^^

* Added ``MeshCfg.edge_refinement``, defaulting to ``4.0``, to control surface mesh resolution for deformable
  primitives and the automatically generated tetrahedral mesh resolution for closed volume deformables. It is ignored
  when ``deformable_props`` is None, since rigid primitive collision approximations are invariant to surface
  subdivision.
