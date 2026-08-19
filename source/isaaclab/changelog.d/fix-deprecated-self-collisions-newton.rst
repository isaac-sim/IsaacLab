Fixed
^^^^^

* Fixed the deprecated :class:`~isaaclab_physx.sim.schemas.ArticulationRootPropertiesCfg` /
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg` ``enabled_self_collisions``
  field silently no-oping under Newton. The legacy writer only authored
  ``physxArticulation:enabledSelfCollisions`` (via ``PhysxArticulationAPI``); Newton's schema
  resolver checks the native ``newton:selfCollisionEnabled`` attribute first and only falls back to
  the PhysX one when it is unauthored, so the value never reached Newton simulations.
  :meth:`~isaaclab.sim.schemas.modify_articulation_root_properties` now also mirrors
  ``enabled_self_collisions`` onto ``newton:selfCollisionEnabled`` (applying
  ``NewtonArticulationRootAPI``), so the deprecated cfg controls self-collisions on both backends.
