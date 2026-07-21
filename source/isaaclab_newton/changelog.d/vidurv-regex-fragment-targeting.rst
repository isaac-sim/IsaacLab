Changed
^^^^^^^

* Changed :func:`~isaaclab_newton.sim.schemas.apply_mujoco_fixed_tendon` to author on
  the given prim only. Target selection, including subtree matching via prim path
  expressions, is now owned by the core family writer
  :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`; pass
  ``f"{prim_path}/**"`` to it to reach descendant ``MjcTendon`` prims.
