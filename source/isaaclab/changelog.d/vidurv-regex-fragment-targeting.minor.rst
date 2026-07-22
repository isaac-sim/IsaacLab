Added
^^^^^

* Added a trailing ``**`` token to prim path expressions in
  :func:`~isaaclab.sim.utils.find_matching_prims`, matching a prim and all of its
  descendants at any depth (instance proxies included).
* Added ``create_if_missing`` to the fragment schema writers and matching spawner
  configuration flags (``mass_props_create_if_missing``,
  ``articulation_props_create_if_missing``, ``joint_drive_props_create_if_missing``)
  to apply a defining USD API to matched prims that do not carry it.

Changed
^^^^^^^

* **Breaking:** Changed the fragment schema writers (e.g.
  :func:`~isaaclab.sim.schemas.apply_rigid_body_properties`) to take a prim path
  expression instead of traversing the subtree of the input prim. A bare prim path now
  matches only that prim; pass ``f"{prim_path}/**"`` to recover the previous
  subtree-wide behavior. Zero matched targets now log a warning and return ``False``
  instead of raising ``ValueError`` on an invalid path.
* **Breaking:** Changed the spawner configuration fragment fields
  (e.g. :attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.rigid_props`,
  :attr:`~isaaclab.sim.spawners.from_files.FileCfg.articulation_props`) to take a
  mapping from target pattern to fragment list instead of a bare fragment or fragment
  list. Keys are per-level regular expressions relative to the spawn prim, a trailing
  ``**`` token matches a prim and all of its descendants, and ``""`` targets the spawn
  prim itself; entries apply in insertion order, so on overlapping targets later
  entries override earlier ones per attribute. Pass ``{"**": [...]}`` to recover the
  previous fragment-list behavior. Legacy single-cfg values are unaffected.
* **Breaking:** Changed the fragment schema writers to no longer apply their defining
  USD API implicitly on bare prims; pass ``create_if_missing=True`` instead. For
  articulation roots, anchoring on the spawn prim now requires a ``{"": [...]}``
  entry in :attr:`~isaaclab.sim.spawners.from_files.FileCfg.articulation_props`
  together with ``articulation_props_create_if_missing=True`` on the spawner
  configuration.
* **Breaking:** Changed :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`
  to author on every matched articulation root, warning when they nest, instead of
  silently pruning nested roots; asset validity is the author's responsibility.
