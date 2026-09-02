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

* Added :attr:`~isaaclab.sim.schemas.SchemaFragment._usd_multi_apply_schema`, which lets a fragment
  declare the multiple-apply USD schema it configures. The default applier writes every applied
  instance and takes each attribute name from USD, so a fragment no longer needs a custom applier to
  reach a multi-instance schema.

Fixed
^^^^^

* Fixed legacy fixed- and spatial-tendon property writers authoring settings outside the schema-declared
  ``physxTendon`` namespace.
