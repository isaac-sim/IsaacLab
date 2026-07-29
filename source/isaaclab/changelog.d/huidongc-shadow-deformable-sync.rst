Added
^^^^^

* Added :class:`~isaaclab.scene_data.SceneDataFormat.Points`, deformable discovery helpers in
  :mod:`isaaclab.scene_data.deformable_discovery`, and geometry mapping on
  :class:`~isaaclab.scene_data.SceneDataProvider` for PhysX/OVPhysX nodal position sync.

Changed
^^^^^^^

* Changed :meth:`~isaaclab.scene_data.SceneDataProvider.get_points` to copy mapped entity
  slices without converting geometry mappings to Python lists on the hot path.

Fixed
^^^^^

* Fixed visual mesh vertex counts in :func:`~isaaclab.scene_data.deformable_discovery.build_deformable_vertex_count_lookup`
  when simulation and render meshes differ in size.
* Fixed deformable USD discovery on in-memory stages by passing the owning stage to child
  prim queries and reading explicit ``apiSchemas`` metadata when composed schemas are absent.
* Fixed :meth:`~isaaclab.scene_data.SceneDataProvider.get_points` overflowing consumer
  particle buffers by clamping entity copies to destination capacity.
