Changed
^^^^^^^

* Deprecated :attr:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg.gravity_compensation_scale`
  in favour of :attr:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg.gravcomp`. Forwarded
  via ``__post_init__``. Removal in 5.0.
* Deprecated :attr:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg.gravity_compensation`
  in favour of :attr:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg.actuatorgravcomp`.
  Forwarded via ``__post_init__``. Removal in 5.0.
* Relocated :class:`MujocoRigidBodyPropertiesCfg` and :class:`MujocoJointDrivePropertiesCfg`
  to :mod:`isaaclab_newton.sim.schemas`. Forwarding shims on :mod:`isaaclab.sim.schemas` and
  :mod:`isaaclab.sim` preserve existing imports. Shims scheduled for removal in 5.0.
