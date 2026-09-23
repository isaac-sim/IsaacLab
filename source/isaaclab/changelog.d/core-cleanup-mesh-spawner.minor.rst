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
