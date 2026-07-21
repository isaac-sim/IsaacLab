Changed
^^^^^^^

* **Breaking:** Removed the opt-in Newton-native actuator path from the PhysX
  :class:`~isaaclab_physx.assets.Articulation`, including the
  ``write_actuator_stiffness_to_sim`` and ``write_actuator_damping_to_sim`` methods that
  only served it. PhysX always uses the Isaac Lab actuator models; remove
  ``use_newton_actuators`` from your ``SimulationCfg`` and switch to the Newton physics
  backend if you need Newton-native actuators.
