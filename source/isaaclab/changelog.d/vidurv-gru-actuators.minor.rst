Added
^^^^^

* Added :class:`~isaaclab.actuators.ActuatorNetGRU` and
  :class:`~isaaclab.actuators.ActuatorNetGRUCfg`, an explicit actuator whose GRU
  network predicts the total joint effort from the joint position, position error, and
  velocity, with optional input and output normalization.
* Added :class:`~isaaclab.actuators.ActuatorNetGRUResidual` and
  :class:`~isaaclab.actuators.ActuatorNetGRUResidualCfg`, an implicit-PD actuator that
  adds a GRU-predicted residual feed-forward effort, with optional input and output
  normalization.
