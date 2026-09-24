Added
^^^^^

* Added :class:`~isaaclab.sim.MeshFileCfg` and :func:`~isaaclab.sim.spawn_from_mesh` to spawn a mesh from a
  mesh file (e.g. ``.obj``, ``.stl``, ``.fbx``) or from in-memory triangle data
  (:class:`~isaaclab.sim.MeshFileCfg.TriangleMeshCfg` or a :class:`trimesh.Trimesh` through
  :class:`~isaaclab.sim.MeshFileCfg.TrimeshObjectCfg`), with optional collision, mesh collision
  approximation, rigid body, mass, and material properties.

Changed
^^^^^^^

* Changed :func:`~isaaclab.terrains.utils.create_prim_from_mesh` to spawn the terrain mesh with
  :class:`~isaaclab.sim.MeshFileCfg`. The authored USD is unchanged, except that the ``translation`` and
  ``orientation`` keyword arguments now apply to the root prim instead of its ``mesh`` child prim; the
  resulting world pose is the same.

Fixed
^^^^^

* Fixed the built-in :class:`~isaaclab.terrains.MeshRepeatedBoxesTerrainCfg`,
  :class:`~isaaclab.terrains.MeshRepeatedCylindersTerrainCfg`, and
  :class:`~isaaclab.terrains.MeshRepeatedPyramidsTerrainCfg` raising ``ValueError`` with their default
  ``object_type``. The resolvable ``"module:function"`` default was looked up as ``make_<object_type>``
  instead of being called.
* Fixed :func:`~isaaclab.sim.schemas.modify_articulation_root_properties` looking up the existing fixed
  joint on the current stage instead of the ``stage`` argument when ``fix_root_link`` is set.
* Fixed :func:`~isaaclab.sim.utils.queries.find_global_fixed_joint_prim` raising ``AttributeError`` for an
  :class:`pxr.Sdf.Path` argument.
* Fixed :class:`~isaaclab.sim.converters.MeshConverter` raising ``ValueError`` for mesh file names with
  more than one dot.
