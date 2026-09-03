Changed
^^^^^^^

* Changed ``MeshCuboidCfg.edge_refinement``, added in 15.6.0 and now generalized to ``MeshCfg.edge_refinement``, to
  apply only when ``deformable_props`` is set. Rigid primitives are no longer subdivided; their surface and collision
  approximation are unaffected, since subdivision only inserted coplanar vertices. Callers relying on a denser rigid
  visual mesh must supply their own mesh asset.

Removed
^^^^^^^

* Removed ``MeshRectangleCfg.resolution``. Deformable callers must use ``MeshCfg.edge_refinement`` to bound surface
  edge length relative to the bounding-box diagonal. Rigid rectangles are now spawned as two triangles.
