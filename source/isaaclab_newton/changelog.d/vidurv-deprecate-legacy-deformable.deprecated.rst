* Deprecated :class:`~isaaclab_newton.sim.schemas.NewtonDeformableBodyPropertiesCfg` in favor of
  an empty deformable slot on the spawner. It now raises a ``DeprecationWarning`` on instantiation
  and will be removed in 3.2. The class carries no fields: replace
  ``deformable_props=NewtonDeformableBodyPropertiesCfg()`` with ``surface_deformable_props=[]``
  when ``physics_material`` is a surface deformable material and with
  ``volume_deformable_props=[]`` otherwise, which is the type the legacy ``deformable_props`` field
  derived. The active physics backend now selects Newton's deformable schemas; the legacy cfg
  selected them itself.
