Changed
^^^^^^^

* **Breaking:** Removed ``use_newton_actuators`` from :class:`~isaaclab.sim.SimulationCfg`.
  The Newton backend now always uses Newton-native actuators (stepped by the physics
  engine) and the PhysX backend always uses the Isaac Lab actuator models. Remove the
  flag from your ``SimulationCfg``; no other migration is needed.
* Changed :func:`~isaaclab.sim.schemas.define_actuator_properties` to gate ``NewtonActuator``
  USD authoring on the active physics backend being Newton instead of the removed
  ``use_newton_actuators`` flag.
