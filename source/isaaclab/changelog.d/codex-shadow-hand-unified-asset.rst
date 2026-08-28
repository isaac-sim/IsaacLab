Added
^^^^^

* Added :class:`~isaaclab.envs.mdp.actions.FixedTendonPositionAction`, an action term that commands
  an articulation's fixed tendons. Tendons have their own index space, so a joint action term cannot
  reach them; pair the two terms to cover a robot whose motors drive both.

* Added :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index`,
  so a tendon position target can be commanded without the caller knowing which physics engine is
  running.


* Added :func:`~isaaclab.sim.schemas.multiple_apply_property_names`, which reports the attribute
  names a multiple-apply API schema writes for a given instance. A multiple-apply schema's property
  namespace prefix is declared by the schema rather than derived from its name, and sibling schemas
  may share one prefix, so the prefix cannot be spelled from the schema name alone.

Fixed
^^^^^

* Fixed legacy fixed- and spatial-tendon property writers authoring settings outside the schema-declared
  ``physxTendon`` namespace.
