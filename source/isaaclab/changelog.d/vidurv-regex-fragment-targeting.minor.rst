Added
^^^^^

* Added a trailing ``**`` token to prim path expressions in
  :func:`~isaaclab.sim.utils.find_matching_prims`, matching a prim and all of its
  descendants at any depth (instance proxies included).
* Added per-family target-pattern fields to spawner configurations
  (e.g. :attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.rigid_props_prim_path`,
  :attr:`~isaaclab.sim.spawners.from_files.FileCfg.articulation_props_prim_path`),
  selecting which prims of a spawned asset receive schema fragments. Patterns are
  relative to the spawn prim; ``None`` keeps the previous behavior.
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
* **Breaking:** Changed the fragment schema writers to no longer apply their defining
  USD API implicitly on bare prims; pass ``create_if_missing=True`` instead. For
  articulation roots, anchoring on the spawn prim now requires setting
  ``articulation_props_prim_path=""`` together with
  ``articulation_props_create_if_missing=True`` on the spawner configuration.
* **Breaking:** Changed :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`
  to raise ``ValueError`` when the expression matches nested articulation roots,
  instead of silently pruning them; author a single root per articulation.
