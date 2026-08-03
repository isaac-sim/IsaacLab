Added
^^^^^

* Added explicit state-buffer advancement so Newton actuator adapters can be
  replayed from backend-owned CUDA graphs.

Changed
^^^^^^^

* Routed Newton articulation actuator setup, compute, reset, and command
  submission through :class:`~isaaclab.actuators.ActuatorCollection`.
