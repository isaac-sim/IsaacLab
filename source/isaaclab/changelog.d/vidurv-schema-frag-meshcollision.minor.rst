Added
^^^^^

* Added the mesh-collision schema-fragment API: the
  :class:`~isaaclab.sim.schemas.MeshCollisionFragment` marker and
  :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg` (carrying the standard
  ``physics:approximation`` token via ``UsdPhysics.MeshCollisionAPI``).
* Added :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`, which applies
  ``UsdPhysics.MeshCollisionAPI`` as the implicit anchor, resolves the
  ``physics:approximation`` token from whichever cooking fragment is present (validated against
  :const:`~isaaclab.sim.schemas.MESH_APPROXIMATION_TOKENS`), and dispatches each fragment via its
  ``func``.

Changed
^^^^^^^

* Changed the mesh-converter ``mesh_collision_props`` slot
  (:attr:`~isaaclab.sim.converters.MeshConverterCfg.mesh_collision_props`) to also accept a list of
  :class:`~isaaclab.sim.schemas.MeshCollisionFragment` fragments. Legacy single cfgs continue to
  work through a transition bridge in the converter.
