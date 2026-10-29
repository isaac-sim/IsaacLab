* Changed the rigid contact buffer of the collision pipeline to the solver's contact capacity for solvers that copy
  the contacts into a buffer of their own (MuJoCo Warp with ``use_mujoco_contacts=False``), unless
  :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.rigid_contact_max` is set larger. Contacts beyond that
  capacity were never used, so the buffer is smaller without changing the simulation.
