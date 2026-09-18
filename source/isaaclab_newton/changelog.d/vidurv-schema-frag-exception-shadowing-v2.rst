Fixed
^^^^^

* Fixed the Newton and MuJoCo schema cfg classes dropping the PhysX routing of the fields they
  inherit from the solver-common base cfgs. Setting ``disable_gravity``, ``contact_offset``, or
  ``rest_offset`` on a Newton or MuJoCo cfg authored a bare ``physics:*`` USD attribute that no
  backend reads instead of the ``physxRigidBody:*`` / ``physxCollision:*`` attribute, and setting
  ``max_joint_velocity`` on :class:`~isaaclab_newton.sim.schemas.NewtonJointDrivePropertiesCfg` or
  :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` raised ``ValueError``. These
  fields now author their PhysX-namespaced attributes on every subclass, matching the base cfgs and
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg`.
