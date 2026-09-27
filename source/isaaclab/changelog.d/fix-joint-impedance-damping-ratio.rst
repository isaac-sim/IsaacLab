Fixed
^^^^^

* Fixed :class:`~isaaclab.controllers.joint_impedance.JointImpedanceController` raising an error when
  :attr:`~isaaclab.controllers.joint_impedance_cfg.JointImpedanceControllerCfg.damping_ratio` is left at its default of ``None``. An
  unset damping ratio now means critical damping (1.0).
* Fixed the ``"variable_kp"`` impedance mode ignoring the configured
  :attr:`~isaaclab.controllers.joint_impedance_cfg.JointImpedanceControllerCfg.damping_ratio` and always using critical damping.
  Configurations that set a ``damping_ratio`` other than 1.0 in this mode now get correspondingly different velocity
  gains; set ``damping_ratio=1.0`` (or leave it unset) to keep the previous critically damped behavior.
