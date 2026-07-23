Changed
^^^^^^^

* Changed :class:`~isaaclab.controllers.JointImpedanceController` to evaluate its impedance law through
  Newton's model-free joint-impedance controller
  (:class:`newton.controllers.ControllerJointImpedanceModelFree`) while preserving its public
  configuration, command, and output contracts. Solves now use float32 internal buffers.

Fixed
^^^^^

* Fixed per-DOF gain-limit clamping in :meth:`~isaaclab.controllers.JointImpedanceController.set_command`
  for the ``variable`` and ``variable_kp`` impedance modes, which previously indexed the robot dimension
  and raised for any joint count other than two.
