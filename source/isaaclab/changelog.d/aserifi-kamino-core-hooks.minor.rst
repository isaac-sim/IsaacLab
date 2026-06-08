Added
^^^^^

* Added :attr:`~isaaclab.actuators.ActuatorBase.route_torque_to` to select whether
  actuator torque is written to Newton ``Control.joint_f`` or ``Control.joint_act``,
  enabling backward-Euler joint treatment under the Kamino solver.

Fixed
^^^^^

* Fixed stale reset poses for maximal-coordinate solvers (e.g. Kamino) by refreshing
  kinematics and derived buffers after in-step resets in
  :meth:`~isaaclab.envs.ManagerBasedRLEnv.step` (no-op for other backends).
