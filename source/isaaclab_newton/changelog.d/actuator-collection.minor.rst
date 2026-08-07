Added
^^^^^

* Added explicit state-buffer advancement so Newton actuator adapters can be
  replayed from backend-owned CUDA graphs.

Changed
^^^^^^^

* Routed Newton articulation actuator setup, compute, reset, and command
  submission through :class:`~isaaclab.actuators.ActuatorCollection`.
* Changed Newton actuator execution on PhysX to aggregate structurally
  compatible joints while retaining their per-joint parameters.
