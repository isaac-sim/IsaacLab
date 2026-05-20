Added
^^^^^

* Added backend-neutral deformable asset APIs, including
  :class:`~isaaclab.assets.DeformableObject`,
  :class:`~isaaclab.assets.DeformableObjectCfg`, and shared base/data classes.
* Added deformable body spawner, schema, and material APIs under
  :mod:`isaaclab.sim`, including
  :class:`~isaaclab.sim.DeformableObjectSpawnerCfg`,
  :class:`~isaaclab.sim.DeformableBodyPropertiesCfg`,
  :func:`~isaaclab.sim.define_deformable_body_properties`,
  :func:`~isaaclab.sim.modify_deformable_body_properties`,
  :class:`~isaaclab.sim.DeformableBodyMaterialCfg`,
  :class:`~isaaclab.sim.SurfaceDeformableBodyMaterialCfg`, and
  :func:`~isaaclab.sim.spawn_deformable_body_material`.
* Added ``pytetwild`` as a package dependency for tetrahedral mesh generation.
* Added deformable API, migration, and tutorial documentation for
  backend-neutral imports and Newton backend selection.

Changed
^^^^^^^

* Changed deformable demos and tutorials to use the backend-neutral
  :mod:`isaaclab.assets` and :mod:`isaaclab.sim` APIs with selectable PhysX or
  Newton backends.
* Changed USD spawning to support deformable objects whose USD assets contain
  embedded tetrahedral mesh data.
* **Breaking:** Renamed :class:`~isaaclab.sim.spawners.meshes.MeshSquareCfg`
  to :class:`~isaaclab.sim.spawners.meshes.MeshRectangleCfg` and
  :func:`~isaaclab.sim.spawners.meshes.spawn_mesh_square` to
  :func:`~isaaclab.sim.spawners.meshes.spawn_mesh_rectangle`. The ``size``
  attribute is now a ``tuple[float, float]`` of X- and Y-axis lengths instead
  of a single edge length. Migrate ``MeshSquareCfg(size=s)`` to
  ``MeshRectangleCfg(size=(s, s))``.
