Added
^^^^^

* Added OVPhysX backend support to :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_collider_offsets`.
  The term previously fell through to the PhysX implementation on OVPhysX and raised ``AttributeError``
  during environment construction, because :class:`~isaaclab_ov.sim.views.OvPhysxView` has none of the
  PhysX ``root_view`` offset accessors. Rest and contact offsets are now written per collision shape through
  the asset's view. An unrecognised physics manager now raises ``ValueError`` naming the backend instead of
  silently selecting the PhysX implementation.
