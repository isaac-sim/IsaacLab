Changed
^^^^^^^

* Changed the tendon fragment functions (e.g. the ``func`` behind
  :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg`) to author on the given prim
  only. Target selection, including subtree matching via prim path expressions, is now
  owned by the core family writers such as
  :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`; pass
  ``f"{prim_path}/**"`` to those writers to reach descendant tendon prims.
