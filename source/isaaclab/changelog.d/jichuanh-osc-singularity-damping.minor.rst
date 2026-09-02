Added
^^^^^

* Added :attr:`~isaaclab.controllers.OperationalSpaceControllerCfg.inertial_decoupling_method` and
  :attr:`~isaaclab.controllers.OperationalSpaceControllerCfg.inertial_decoupling_params` to regularize
  the task-space inertia inversion near kinematic singularities. Setting ``"cond_clamp"`` bounds the
  condition number of :math:`J M^{-1} J^T`, keeping command forces finite where the default ``"inv"``
  diverges to unbounded torques and eventually ``NaN``. This most often affects non-redundant (6-DoF)
  arms such as the UR10, but redundant arms reach such configurations too.
