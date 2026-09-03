Fixed
^^^^^

* Fixed the Newton and MuJoCo schema cfg classes cancelling the PhysX routing of fields they
  inherit from the solver-common base classes. Redeclaring ``_usd_field_exceptions`` as an empty
  dict shadowed the base class's routing table, so setting ``disable_gravity``,
  ``contact_offset``, or ``rest_offset`` on a Newton or MuJoCo cfg authored a bare ``physics:*``
  attribute that no backend reads, and setting ``max_joint_velocity`` on
  :class:`~isaaclab_newton.sim.schemas.NewtonJointDrivePropertiesCfg` or
  :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` raised ``ValueError``.
  These fields now route to their PhysX namespaces on every subclass, matching the base classes
  and :class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg`, which already
  inherited the routing correctly.
