Fixed
^^^^^

* Fixed the ``"variable_kp"`` impedance mode of :class:`~isaaclab.controllers.joint_impedance.JointImpedanceController`
  ignoring :attr:`~isaaclab.controllers.joint_impedance_cfg.JointImpedanceControllerCfg.damping_ratio` and always
  using critical damping, matching the ``"variable_kp"`` mode of
  :class:`~isaaclab.controllers.operational_space.OperationalSpaceController`. Configurations that set a
  ``damping_ratio`` other than 1.0 in this mode now get correspondingly different velocity gains; use the default
  ``damping_ratio=1.0`` to keep the previous critically damped behavior.
