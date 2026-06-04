Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyCfg`, the ``mjc:*`` single-namespace
  rigid-body fragment (``mjc:gravcomp``) for Newton's MuJoCo solver. It composes with
  :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg` and
  :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg` in a ``rigid_props`` fragment list.
