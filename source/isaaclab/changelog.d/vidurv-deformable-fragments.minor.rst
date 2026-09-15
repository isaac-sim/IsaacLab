Added
^^^^^

* Added the deformable-body fragment families: :class:`~isaaclab.sim.schemas.DeformableBodyFragment`
  and :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg` with the expression-targeted
  writers :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties` and
  :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties`.
* Added deformable material fragments
  (:class:`~isaaclab.sim.spawners.materials.OmniPhysicsDeformableMaterialCfg`,
  :class:`~isaaclab.sim.spawners.materials.OmniPhysicsSurfaceDeformableMaterialCfg`) under the new
  :class:`~isaaclab.sim.spawners.materials.DeformableMaterialFragment` marker, accepted by the
  ``physics_material`` spawner fields.
* Added ``volume_deformable_props`` and ``surface_deformable_props`` mappings to
  :class:`~isaaclab.sim.spawners.DeformableObjectSpawnerCfg`, applying deformable fragments by
  target pattern relative to the spawn prim.
* Added :meth:`~isaaclab.physics.PhysicsManager.setup_deformable_body` so each physics backend
  applies its own deformable anchor schemas.
* Added ``tetrahedralization_edge_length_fac`` to
  :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties` and
  :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties`, so the fragment writers
  control the automatic tetrahedralization target edge length like
  :func:`~isaaclab.sim.schemas.define_deformable_body_properties` does.

Changed
^^^^^^^

* Changed :attr:`~isaaclab.sim.spawners.meshes.MeshCfg.edge_refinement` to apply to the
  ``volume_deformable_props`` and ``surface_deformable_props`` slots as well, not only the legacy
  ``deformable_props`` field.
* Changed physics-material fragment lists to allow mixing rigid-body and deformable material
  fragments on one material prim; the ``UsdPhysics.MaterialAPI`` anchor is applied only when a
  rigid-body fragment is present. Renamed ``spawn_rigid_body_material_from_fragments`` to
  :func:`~isaaclab.sim.spawners.materials.spawn_physics_material_from_fragments` accordingly.
