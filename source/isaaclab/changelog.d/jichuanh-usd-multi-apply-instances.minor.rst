Added
^^^^^

* Added :func:`~isaaclab.sim.schemas.apply_multi_apply`, the applier for fragments over a multiple-apply
  USD schema. The fragment names the schema in ``_usd_applied_schema`` and the attribute namespace in
  ``_usd_namespace``, as other fragments do, and its required ``instance_names`` field selects the
  instances to tune: a name, a list of names, or ``None`` for every instance on the prim.

* Added :func:`~isaaclab.sim.schemas.resolve_applied_schema_instances`, which returns the instance
  names of one multiple-apply API schema among a prim's applied schemas.

Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.schemas.modify_fixed_tendon_properties` and
  :func:`~isaaclab.sim.schemas.modify_spatial_tendon_properties` writing to a namespace PhysX never
  reads. The writers spelled ``<Schema>:<instance>:<property>``, but a multiple-apply schema declares
  its own property namespace (``physxTendon:``) rather than deriving it from the class name, so every
  write landed on a custom attribute nothing consumes. The spatial writer now also writes attachment
  roots only, since leaf attachments do not declare the tendon's dynamics.
