Added
^^^^^

* Added deformable nodal position export to ``OvPhysxSceneDataBackend`` for shadow Newton
  visualization sync.

Fixed
^^^^^

* Fixed deformable geometry counts falling back to padded ``max_simulation_nodes_per_body``
  when OVPhysX view prim paths did not exactly match discovered deformable roots.
