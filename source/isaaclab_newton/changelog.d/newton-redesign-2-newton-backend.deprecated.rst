* Deprecated ``NewtonManager.get_model``, ``get_state_0``, ``get_state_1``, ``get_control``, ``get_contacts`` and
  ``get_solver``; read ``manager.backend.model`` and the other backend fields instead.
* Deprecated ``NewtonManager.cl_register_site``, ``activate_newton_actuator_path`` and
  ``register_pre_actuator_callback`` in favor of ``register_site``, ``activate_actuators`` and
  ``register_control_callback``.
