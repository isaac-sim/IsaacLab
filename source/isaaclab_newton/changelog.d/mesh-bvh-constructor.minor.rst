Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonCfg.mesh_bvh_constructor` to select the triangle-mesh BVH constructor.
* Added current Newton contact-friction, velocity-limit, velocity-pass, scheduling, drive, kernel-tuning, and
  row-watermark options to :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg`.
* Added :meth:`~isaaclab_newton.physics.MJWarpSolverCfg.validate_contact_mode` so standalone and coupled solver
  managers share contact-pipeline validation owned by the solver configuration.

Deprecated
^^^^^^^^^^

* Deprecated setting :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_contacts` to ``True``, which selects
  MuJoCo's internal contact pipeline. Set it to ``False`` and configure
  :attr:`~isaaclab_newton.physics.NewtonCfg.collision_cfg` instead.
