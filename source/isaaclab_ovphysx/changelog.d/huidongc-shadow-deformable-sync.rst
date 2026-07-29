Added
^^^^^

* Added deformable nodal position export to ``OvPhysxSceneDataBackend`` for shadow Newton
  visualization sync.

Fixed
^^^^^

* Fixed deformable geometry counts falling back to padded ``max_simulation_nodes_per_body``
  when OVPhysX view prim paths did not exactly match discovered deformable roots.
* Fixed ``OvPhysxSceneDataBackend`` surface/volume deformable view construction to pass the
  required OVPhysX tensor-type arguments, restoring nodal position export used by shadow
  Newton / OVRTX cloth rendering.
