* Changed the collision pipeline to be built once with the solver's contact capacity as its rigid contact capacity,
  for solvers that copy the contacts into a buffer of their own (MuJoCo Warp with ``use_mujoco_contacts=False``),
  unless :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.rigid_contact_max` is set larger, instead of
  growing the contacts after construction. The solver copies at most its capacity, so the simulation is unchanged
  while the contacts fit it.
