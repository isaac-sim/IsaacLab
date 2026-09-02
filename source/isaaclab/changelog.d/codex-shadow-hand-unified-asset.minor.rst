Added
^^^^^

* Added :class:`~isaaclab.envs.mdp.actions.FixedTendonPositionAction`, an action term that commands
  an articulation's fixed tendons. Tendons have their own index space, so a joint action term cannot
  reach them; pair the two terms to cover a robot whose motors drive both.

* Added :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index`,
  so a tendon position target can be commanded without the caller knowing which physics engine is
  running.

* Added :func:`~isaaclab.sim.schemas.multiple_apply_property_name`, which reports the attribute
  name a multiple-apply API schema writes for one property of one instance. The namespace prefix is
  declared by the schema rather than derived from its name and may be shared by sibling schemas, and
  the instance is not always the last namespace, so the name cannot be spelled from the schema name.

* Added :func:`~isaaclab.sim.schemas.resolve_applied_schema_instances`, which returns the instance
  names of one multiple-apply API schema among a prim's applied schemas.

Fixed
^^^^^

* Fixed legacy fixed- and spatial-tendon property writers authoring settings outside the schema-declared
  ``physxTendon`` namespace.
